# MPIB NetPlugin Implementation Design

## 1. Executive Summary

The MPIB (Multi-Path IB) plugin enables heterogeneous multi-rail data transfer for NCCL by decoupling the **mechanism** (plugin) from the **policy** (external agent). It exposes a single logical device to NCCL (`ndev=1`) while internally managing traffic across two physical NICs (NIC0, NIC1).

**Key Features:**

* **Static Fusion:** Hides physical topology from NCCL; reports aggregate bandwidth.
* **Agent-Driven Routing:** (Phase 4) A sidecar daemon controls traffic splits via shared memory (weights 0.0–1.0).
* **Dynamic Splitting:** Supports both whole message WR and fragmentation based on weights.
* **Multi-QP Support:** Configurable number of QPs per physical NIC.

**Current Status (vs. Design):**

| Component | Status | Notes |
|-----------|--------|-------|
| Single-NIC data path | ✅ Done | net_ib fork working with `NCCL_IB_HCA` |
| Multi-NIC enumeration | ✅ Done | `MPIB_HCA_SOUT` / `MPIB_HCA_SUP` env vars |
| Multi-rail QP setup | ✅ Done | 2 QPs per connection (one per NIC) |
| Static splitting | ✅ Done | 50/50 split works; uses net_ib-style per-QP IMM |
| SRQ + dynamic rail skipping | ✅ Done | Per-device SRQ; inactive rails skipped entirely; see [dual-rail-cqe.md](dual-rail-cqe.md) |
| GDR (nv_peermem) | ✅ Done | `NCCL_PTR_CUDA` advertised; GPU MR via `ibv_reg_mr` transparent |
| Relaxed ordering | ✅ Done | `ibv_reg_mr_iova2` + `IBV_ACCESS_RELAXED_ORDERING`, default on |
| Flush (`iflush`) | ✅ Done (no-op) | PCIe ordering makes flush unnecessary — see §9 |
| Agent integration | ❌ Phase 4 | Deferred |

### Notes: Multi-Rail Correctness

MPIB posts a signaled `RDMA_WRITE_WITH_IMM` on each **active** rail only.
Inactive rails are skipped entirely (no 0-byte IMMs). The receiver uses a
per-device SRQ and learns the active rail set from `imm_data` (mask-learning
protocol). SEND completion uses `events[devIndex]`; RECV completion uses
`expected_mask == seen_mask`. See [dual-rail-cqe.md](dual-rail-cqe.md) for
the full SRQ design.

---

## 2. Architecture & Components

### A. The Plugin (Mechanism)

Stateless executor.

* **Init:** Enumerates 2 NICs (via `MPIB_HCA_SOUT` / `MPIB_HCA_SUP`), builds merged vDev.
* **Connect:** Establishes QPs on *both* NICs for every connection.
* **Data Path:** Reads policy (static weight for bring-up) -> Fragments data -> Posts to QPs.
* **Progress:** Polls all CQs -> Aggregates completions.

### B. The Agent (Policy) — *Phase 4*

Intelligent controller (outside plugin scope).

* **Role:** Monitoring topology, congestion, and link health.
* **Output:** Updates the `mpib_policy_shm` table.
* **Example Policies:**
  * *Singlepath:* `Rank_X: {NIC1_Weight=1.0}`
  * *Multipath:* `Rank_Y: {NIC1_Weight=0.5}`

### C. Shared Memory Interface — *Phase 4*

Read-only for Plugin, Read-Write for Agent.

```c
struct mpib_policy_entry {
    _Atomic float nic1_weight; // 0.0 (All NIC0) to 1.0 (All NIC1)
    _Atomic uint32_t version;  // For coherency checks
};

struct mpib_policy_shm {
    uint32_t magic;
    struct mpib_policy_entry peers[MAX_RANKS]; 
};
```

---

## 3. Detailed Design

### I. Device Management & Implicit Binding

We bypass `makeVDevice`. The binding is implicit: **Logical Device 0 ≡ All Physical Devices**.

* **`init()`**:
  * Reads `MPIB_HCA_SOUT` (scaleout NIC, e.g., `"mlx5_3"`) and `MPIB_HCA_SUP` (scaleup NIC, e.g., `"mlx5_4"`).
  * Opens both HCAs: `mpibDevs[0]` = SOUT, `mpibDevs[1]` = SUP.
  * Creates single merged vDev: `mpibMergedDevs[0].vProps = {ndevs=2, devs=[0,1]}`.
* **`devices()`**: Returns `1` (single fused vDev).
* **`getProperties(0)`**:
  * Returns `maxSpeed = mpibDevs[0].speed + mpibDevs[1].speed`.
  * Returns `pciPath = mpibDevs[0].pciPath` (SOUT NIC).
  * Returns `vProps = {ndevs=2, devs=[0,1]}`.
  * Returns `ptrSupport = NCCL_PTR_HOST | NCCL_PTR_CUDA` (when nv_peermem loaded).
  * Returns `forceFlush = 0` (no-op flush; PCIe ordering sufficient — see §9).

### II. Connection Establishment

**Existing structures (from current code):**

```c
struct mpibConnectionMetadata {
    struct mpibQpInfo qpInfo[MPIB_IB_MAX_QPS];   // per-QP: qpn, ece, devIndex
    struct mpibDevInfo devs[MPIB_IB_MAX_DEVS_PER_NIC]; // per-rail: gid, rkey, mtu
    uint64_t addr;   // CTS fifo base
    int ndevs;       // number of rails
    int tc, sl;      // traffic class / service level
};
```

**`connect(dev=0, handle)` Logic (current impl):**

1. Exchange `vProps` (local/remote rail counts).
2. Read `MPIB_SOUT_QP` (default 1) and `MPIB_SUP_QP` (default 1) for per-device QP counts.
3. `nqps = nqpsSout + nqpsSup` — total QPs across both devices.
4. QP array layout is **contiguous by device**: `[SOUT_0, ..., SOUT_{n-1}, SUP_0, ..., SUP_{m-1}]`
5. For each QP index `q in [0, nqps)`:
   * `devIndex = (q < nqpsSout) ? 0 : 1`
   * Create QP on `mpibDevs[devIndex]`
   * Store `qp->devIndex = devIndex`
6. Exchange metadata (includes `nqpsSout`, `nqpsSup` for validation), call `mpibRtrQp` / `mpibRtsQp`.
7. Map `remDevIdx` for striping remote rkeys.

### III. QP Selection & Striping Strategy

**Overview:**

MPIB uses a two-level data distribution strategy:
1. **Device split**: Each message is divided between SOUT (dev0) and SUP (dev1)
2. **QP round-robin**: Within each device, QPs are selected in round-robin fashion across requests

**Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MPIB_SOUT_QP` | 1 | Number of QPs on SOUT device |
| `MPIB_SUP_QP` | 1 | Number of QPs on SUP device |

**QP Array Layout:**

With `MPIB_SOUT_QP=2` and `MPIB_SUP_QP=3`, the QP array is:
```
Index:    [0]      [1]      [2]      [3]      [4]
QP:       SOUT_0   SOUT_1   SUP_0    SUP_1    SUP_2
devIndex:    0        0        1        1        1
```

**Per-Request QP Selection:**

Each `isend`/`irecv` uses exactly **`ndevs` QPs** (one from each device). The selection uses `fifoHead` as a round-robin counter:

```c
// For device 0 (SOUT):
qpIndex = fifoHead % nqpsSout

// For device 1 (SUP):
qpIndex = nqpsSout + (fifoHead % nqpsSup)
```

**Example with SOUT_QP=2, SUP_QP=3:**

| fifoHead | SOUT QP | SUP QP |
|----------|---------|--------|
| 0 | qps[0] (0%2=0) | qps[2] (2+0%3=2) |
| 1 | qps[1] (1%2=1) | qps[3] (2+1%3=3) |
| 2 | qps[0] (2%2=0) | qps[4] (2+2%3=4) |
| 3 | qps[1] (3%2=1) | qps[2] (2+3%3=2) |
| 4 | qps[0] (4%2=0) | qps[3] (2+4%3=3) |
| 5 | qps[1] (5%2=1) | qps[4] (2+5%3=4) |

This ensures all QPs in each device pool are utilized evenly across requests.

**CTS QP Selection:**

The receiver's CTS (Clear-To-Send) RDMA_WRITE uses a **fixed QP per device** (not round-robin). This matches net_ib's design:

```c
uint32_t dev = fifoHead % ndevs;  // Alternates 0, 1, 0, 1, ...
// CTS always uses the first QP on the selected device:
// - dev 0 (SOUT): qps[0]
// - dev 1 (SUP):  qps[nqpsSout]
```

**Why not round-robin for CTS?** CTS is control-plane traffic (small, frequent). Pinning it to one QP per device enables the simple signaling rule `slot == devIndex`, which guarantees each CTS QP gets periodic signaled completions. Round-robin CTS across multiple QPs would require complex per-QP sequence tracking to avoid send queue overflow.

**CTS Signaling:**

```c
if (slot == ctsQp->devIndex) {
    wr.send_flags |= IBV_SEND_SIGNALED;
}
```

With 2 devices, device 0's CTS QP signals at slots 0, 2, 4, ... and device 1's CTS QP signals at slots 1, 3, 5, ... ensuring both queues drain.

**Key Implementation Functions:**

| Function | Location | Purpose |
|----------|----------|---------|
| `mpibCommBaseGetQpForRequest()` | mpib_common.h | Select QP by device index + round-robin |
| `mpibCommBaseGetNqpsPerRequest()` | mpib_common.h | Returns `ndevs` (QPs used per request) |
| `mpibRecvCommGetQpForCts()` | mpib_p2p.h | Select QP for CTS posting |

### IV. Request Lifecycle (`isend`)

**Current flow:**

1. Wait for CTS slot (`slots[0].idx == fifoHead+1`).
2. Allocate `mpibRequest`, set `type = MPIB_NET_IB_REQ_SEND`.
3. For each device `i in [0, ndevs)`:
   * Call `mpibCommBaseGetQpForRequest(comm, fifoHead, i, &qp, &qpIndex)`
   * Call `mpibAddEvent(req, qp->devIndex)` to track expected completion
4. Build and post WRs for each selected QP (one per device).
5. Increment `fifoHead`, return `*request = req`.

**WR chain structure (per active QP):**

```
wrs[0..nreqs-1]: IBV_WR_RDMA_WRITE (data, not signaled)
wrs[nreqs]:      IBV_WR_RDMA_WRITE_WITH_IMM (signaling WR, signaled)
```

The signaling WR carries `imm_data = (slot | active_mask << 8 | size_q << 10)`. On the leader rail it also writes completion sizes to `remCmplsRecords`; on non-leader rails it is IMM-only (`num_sge=0`).

**Data Split (MPIB-CUSTOM):**

Each request's data is split between devices based on agent hints (128B aligned). The sender computes `active_mask` (bit0=SOUT, bit1=SUP) and **only posts to active rails** — inactive rails are skipped entirely.

**Completion model:**

- **SEND:** `events[devIndex]`-based (one event per active QP).
- **RECV:** SRQ + mask-learning. Receiver learns `expected_mask` from the first arriving IMM; completion when `seen_mask == expected_mask`. See [dual-rail-cqe.md](dual-rail-cqe.md).

### V. Progress Engine (`test`)

`mpibTest()` polls both device CQs. Completion criteria differ by request type:

- **SEND:** `events[0] == 0 && events[1] == 0` (same as net_ib).
- **RECV:** `expected_mask != 0 && seen_mask == expected_mask` (SRQ mask-learning). The `expected_mask` is learned from the first arriving IMM's `active_mask` field; each CQE sets a bit in `seen_mask`.

SRQ refill (`mpibSrqCheckAndRefill`) runs in both `irecv` and `test` to prevent RNR.

---

## 4. Implementation Steps (Revised)

### Phase 1: Single-NIC Baseline ✅ (Current State)

**Status:** Complete. The plugin loads, connects, and transfers data with one HCA.

**What's Working:**
* `init()` opens one HCA via `NCCL_IB_HCA`.
* `devices()` returns 1, `getProperties()` returns correct speed/pciPath.
* `listen/connect/accept` complete TCP+QP handshake.
* `isend/irecv/test` work with CTS FIFO protocol.
* Testbench skeleton exists in `ext-net/mpib/testbench/`.

**Validation:** Use `launch_nccl.sh` with single HCA:

```bash
# In ext-net/mpib/scripts/launch_nccl.sh, set NCCL_IB_HCA=mlx5_3
./scripts/launch_nccl.sh
# Expect: NCCL loads mpib, transfers complete, no errors
```

---

### Phase 2: Dual-NIC Enumeration & Connection

**Goal:** Open two HCAs, create QPs on both, verify connections.

#### Step 2.1: Environment & Init Changes

**New env vars:**
* `MPIB_HCA_SOUT` — Scaleout NIC (e.g., `"mlx5_3"`)
* `MPIB_HCA_SUP` — Scaleup NIC (e.g., `"mlx5_4"`)

**File:** [mpib_init.cc](../src/mpib_init.cc)

**Changes:**

1. Replace `NCCL_IB_HCA` parsing with:
   ```c
   const char *hcaSout = mpibGetEnv("MPIB_HCA_SOUT");
   const char *hcaSup  = mpibGetEnv("MPIB_HCA_SUP");
   if (!hcaSout || !hcaSup) {
       WARN("NET/MPIB : MPIB_HCA_SOUT and MPIB_HCA_SUP are required");
       return ncclInvalidUsage;
   }
   ```

2. Open both HCAs:
   ```c
   // mpibDevs[0] = SOUT
   NCCLCHECK(mpibOpenDevice(hcaSout, &mpibDevs[0]));
   // mpibDevs[1] = SUP
   NCCLCHECK(mpibOpenDevice(hcaSup, &mpibDevs[1]));
   mpibNIbDevs = 2;
   ```

3. Create merged vDev:
   ```c
   mpibMergedDevs[0].vProps.ndevs = 2;
   mpibMergedDevs[0].vProps.devs[0] = 0;  // SOUT
   mpibMergedDevs[0].vProps.devs[1] = 1;  // SUP
   mpibMergedDevs[0].speed = mpibDevs[0].speed + mpibDevs[1].speed;
   mpibNMergedIbDevs = 1;
   ```

**Validation checkpoint:** Update `launch_nccl.sh`:
```bash
# Add to launch_nccl.sh:
MPIB_HCA_SOUT=${MPIB_HCA_SOUT:-mlx5_3}
MPIB_HCA_SUP=${MPIB_HCA_SUP:-mlx5_4}
# ...
-x "MPIB_HCA_SOUT=${MPIB_HCA_SOUT}"
-x "MPIB_HCA_SUP=${MPIB_HCA_SUP}"

# Run:
./scripts/launch_nccl.sh
# Log should show: "Using [0]mlx5_3:SOUT [1]mlx5_4:SUP"
```

#### Step 2.2: Connect/Accept Multi-Rail QP Creation

**File:** [mpib_connect.cc](../src/mpib_connect.cc)

The QP creation loop already iterates over `vProps.ndevs`:

```c
for (int qpIndex = 0; qpIndex < nqps; qpIndex++) {
    int devIndex = qpIndex % comm->base.vProps.ndevs;
    mpibInitCommDevBase(comm->base.vProps.devs[devIndex], ...);
    // creates QP on that device
}
```

**Verify:** With `vProps.ndevs=2` and `IB_QPS_PER_CONNECTION=1`:
* `nqps = 1 * 2 = 2`
* QP0 on devIndex=0 (mlx5_0)
* QP1 on devIndex=1 (mlx5_1)

**Changes needed:**

1. Ensure `mpibInitCommDevBase` is idempotent (called once per devIndex, not per QP).
2. Fix potential double-init: track `comm->devs[devIndex].base.pd != NULL`.

**Validation checkpoint:**

```bash
# In launch_nccl.sh, set NCCL_DEBUG=TRACE
./scripts/launch_nccl.sh
# Should see: mpibCreateQp logs for both SOUT and SUP devices
```

#### Step 2.3: End-to-End Validation with NCCL Tests

**No separate testbench.** Use `launch_nccl.sh` with `nccl-tests` directly.

**File:** [scripts/launch_nccl.sh](../scripts/launch_nccl.sh)

**Required changes:**
```bash
# Replace NCCL_IB_HCA with:
MPIB_HCA_SOUT=${MPIB_HCA_SOUT:-mlx5_3}
MPIB_HCA_SUP=${MPIB_HCA_SUP:-mlx5_4}

# Add to MPIRUN env exports:
-x "MPIB_HCA_SOUT=${MPIB_HCA_SOUT}"
-x "MPIB_HCA_SUP=${MPIB_HCA_SUP}"

# Remove or keep NCCL_IB_HCA as fallback for single-NIC mode
```

**Validation:**

```bash
# Multi-node test with dual-rail
MPIB_HCA_SOUT=mlx5_3 MPIB_HCA_SUP=mlx5_4 ./scripts/launch_nccl.sh

# Verify both NICs are used:
# - Check NCCL_DEBUG output for QP creation on both devices
# - Monitor with: watch -n1 'ethtool -S mlx5_3 | grep tx_bytes; ethtool -S mlx5_4 | grep tx_bytes'
```

**Deliverable:** `launch_nccl.sh` successfully runs `all_reduce_perf` with QPs on both SOUT and SUP NICs.

---

### Phase 3: Static Message Splitting (Bring-Up)

**Goal:** Split data across rails with a fixed weight. Verify correctness.

#### Step 3.1: Hardcoded Split (No Config)

For bring-up, we **hardcode** a split ratio. No environment variable.

Currently: SOUT gets 100%, SUP gets 0%. This can be changed to 50/50 for dual-rail validation.

#### Step 3.2: WR Construction in `mpibIsend`

**File:** [mpib_p2p.cc](../src/mpib_p2p.cc)

**Implementation:**

WR construction is inlined directly in `mpibIsend()`. For each QP, we build a chain
where the last WR is `RDMA_WRITE_WITH_IMM` (signaled), and preceding WRs (if any)
are plain `RDMA_WRITE`.

**Two-level striping logic:**

```c
// Level 1: Device split
sizeSout[r] = reqSize;  // 100% to SOUT for now
sizeSup[r] = 0;         // TODO: configurable split ratio

// Level 2: QP chunk size (128B aligned)
qpChunkSout[r] = DIVUP(DIVUP(sizeSout[r], nqpsSout), 128) * 128;
qpChunkSup[r]  = DIVUP(DIVUP(sizeSup[r], nqpsSup), 128) * 128;

// Per-QP posting loop
for (int i = 0; i < nqps; i++) {
    qp = getQpForRequest(fifoHead, i);
    devIndex = qp->devIndex;  // 0=SOUT, 1=SUP
    
    for (int r = 0; r < nreqs; r++) {
        // Compute offset/length based on devIndex and stripe position
        length = min(devSize - devOffset, chunkSize);
        
        if (r == nreqs - 1) {
            // Last WR: RDMA_WRITE_WITH_IMM, signaled
            if (nreqs > 1) {
                // Write completion sizes to remCmplsRecords
            } else {
                // Write data directly
            }
        } else {
            // Data-only WR
        }
    }
    
    ibv_post_send(qp, wrs);
}
```

**Key detail:** RDMA_WRITE goes directly to receiver buffer at computed offsets. No receiver-side reassembly needed.

#### Step 3.3: Receiver-Side Offset Handling

The receiver's CTS slots already provide `slots[r].addr` as the base. The sender computes:
* NIC0 posts: `remote_addr = slots[r].addr`, `length = sizeNic0`
* NIC1 posts: `remote_addr = slots[r].addr + sizeNic0`, `length = sizeNic1`

No receiver code change needed—RDMA writes directly to user buffer.

#### Step 3.4: Completion Tracking

- **SEND:** One event per active QP via `mpibAddEvent`. `test()` drains `events[]`.
- **RECV:** SRQ-based. Receiver does not post per-QP recv WRs. Instead, generic WRs are posted to per-device SRQs. Completion uses mask-learning from `imm_data`.

**Validation:**

```bash
# Run with dual-rail
MPIB_HCA_SOUT=mlx5_3 MPIB_HCA_SUP=mlx5_4 ./scripts/launch_nccl.sh

# Verify with ethtool counters (run on each node):
watch -n1 'echo "SOUT:"; ethtool -S mlx5_3 | grep tx_bytes; echo "SUP:"; ethtool -S mlx5_4 | grep tx_bytes'
```

---

### Phase 4: Agent Integration (Deferred)

**Goal:** Replace static weight with runtime-updatable shared memory policy.

#### Step 4.1: Shared Memory Setup

**File:** `mpib_agent.h`, `mpib_agent.cc` (new files)

```c
#define MPIB_SHM_NAME "/mpib_policy"
#define MPIB_SHM_MAGIC 0x4D504942

struct mpib_policy_entry {
    _Atomic float nic1_weight;
    _Atomic uint32_t version;
};

struct mpib_policy_shm {
    uint32_t magic;
    uint32_t nranks;
    struct mpib_policy_entry peers[MPIB_MAX_RANKS];
};

ncclResult_t mpibAgentInit(int rank, int nranks);
float mpibAgentGetWeight(int peerRank);
ncclResult_t mpibAgentFinalize(void);
```

#### Step 4.2: Integrate into Data Path

In `mpibMultiSend`, replace the hardcoded 50/50 split with:

```c
float w1 = mpibAgentGetWeight(comm->peerRank);
size_t sizeSup = (size_t)(totalSize * w1);
size_t sizeSout = totalSize - sizeSup;
```

#### Step 4.3: Agent Daemon (Python Prototype)

```python
#!/usr/bin/env python3
import mmap, struct, time

SHM_PATH = "/dev/shm/mpib_policy"
MAGIC = 0x4D504942

def set_weight(rank, weight):
    with open(SHM_PATH, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        # header: magic(4) + nranks(4)
        offset = 8 + rank * 8  # entry: weight(4) + version(4)
        mm[offset:offset+4] = struct.pack('f', weight)
        ver = struct.unpack('I', mm[offset+4:offset+8])[0]
        mm[offset+4:offset+8] = struct.pack('I', ver + 1)
        mm.close()

# Example: toggle weight every 5 seconds
while True:
    set_weight(0, 0.0); time.sleep(5)
    set_weight(0, 1.0); time.sleep(5)
```

#### Step 4.4: Dynamic Split Design

Current implementation uses **design (3)** (fully dynamic, sender decides per-request).
Previously this had ~15% overhead because inactive rails still posted 0-byte IMMs and
matching recv WRs. This was resolved by the SRQ + `active_mask` redesign: the sender
only posts to rails with data, and the receiver learns the active set from `imm_data`.
See [dual-rail-cqe.md](dual-rail-cqe.md).

---

## 5. Testing Matrix

| Test | Command | Expected |
|------|---------|----------|
| Single-NIC baseline | `NCCL_IB_HCA=mlx5_3 ./scripts/launch_nccl.sh` | Pass, single NIC |
| Dual-NIC 50/50 | `MPIB_HCA_SOUT=mlx5_3 MPIB_HCA_SUP=mlx5_4 ./scripts/launch_nccl.sh` | Pass, ~equal traffic on both |
| Agent toggle | (Phase 4) `nccl-tests + agent script` | Traffic shifts dynamically |

---

## 6. Risk & Mitigation

| Risk | Mitigation |
|------|------------|
| Cross-QP ordering with split | Use per-rail IMM (net_ib-style), i.e. one CQE per rail |
| Uneven MTU across NICs | Take `min(mtu)` in connect handshake |
| MR registration per-device | `mpibMrHandle.mrs[]` already per-rail |
| Request tracking overflow | `events[4]` array bounds; assert `devIndex < 4` |

---

## 7. Known Bugs

None currently open. `makeVDevice` returns `ncclInvalidUsage` by design (MPIB manages its own device merging).

---

## 8. Resolved Performance Issues

Both issues below were resolved by the SRQ + dynamic rail skipping redesign (see [dual-rail-cqe.md](dual-rail-cqe.md)).

**Issue #1: Dual-Device Overhead (~15% BW Loss)** — Resolved.
Previously, inactive rails still posted 0-byte `RDMA_WRITE_WITH_IMM` and matching recv WRs. With SRQ, the sender computes `active_mask` and skips inactive rails entirely; the receiver uses mask-learning to know how many CQEs to expect.

**Issue #2: Medium Message Performance Spike (67MB)** — Resolved.
Likely a secondary effect of the per-QP recv posting overhead at that message size. Eliminated by the SRQ refill model.

---

## 9. GDR, Relaxed Ordering & Flush

All three follow net_ib's design with no MPIB-specific changes.

**GDR (nv_peermem):** The `nvidia_peermem` kernel module transparently intercepts `ibv_reg_mr()` for CUDA virtual addresses — no separate code path needed. `mpibGdrSupport()` probes sysfs for the module; if present, `getProperties` advertises `NCCL_PTR_CUDA`.

**Relaxed ordering:** Enabled by default when libibverbs exposes `ibv_reg_mr_iova2` (IBVERBS_1.8). MRs are registered with `IBV_ACCESS_RELAXED_ORDERING` via the `iova2` path. `NCCL_NET_MR_FLAG_FORCE_SO` overrides back to strict ordering. Detection checks the dlsym'd function pointer directly (calling with NULL args segfaults).

**Flush (`iflush`):** No-op — returns `*request = NULL`. On PCIe, CQ completion already guarantees GPU data visibility (strong write ordering through the root complex). On NVLink/NVSwitch topologies the NVLink bridge's write-combining buffer can reorder writes, so a real flush (loopback `IBV_WR_RDMA_READ`) would be required — see [plan-gdrNvpeermem.prompt.md](../.github/prompts/plan-gdrNvpeermem.prompt.md) steps 4–6. In practice, mlx5 hardware also preserves ordering to GPU BARs even with RO enabled, though this is not a spec guarantee.
