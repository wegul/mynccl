# MPIB NetPlugin Implementation Design

## 1. Executive Summary

The MPIB (Multi-Path IB) plugin enables heterogeneous multi-rail data transfer for NCCL by decoupling the **mechanism** (plugin) from the **policy** (external agent). It exposes a single logical device to NCCL (`ndev=1`) while internally managing traffic across two physical NICs (NIC0, NIC1).

**Key Features:**

* **Static Fusion:** Hides physical topology from NCCL; reports aggregate bandwidth.
* **Path Isolation:** Topology-aware classification suppresses SUP QPs when the SUP fabric is unreachable (inter-island vanilla). See [path_isolation.md](path_isolation.md).
* **Dynamic Splitting:** Per-transfer SOUT/SUP split via `mpibGetSupBw()`. Integer-only arithmetic (no floats on hot path).
* **Multi-QP Support:** Configurable QP count per rail (`MPIB_SOUT_QP`, `MPIB_SUP_QP`).
* **Agent-Ready:** SHM hint interface (`mpib_agent_iface.h`) for future agent-driven multipath (advanced mode).

**Current Status (vs. Design):**

| Component | Status | Notes |
|-----------|--------|-------|
| Multi-NIC enumeration | ✅ Done | `MPIB_HCA_SOUT` / `MPIB_HCA_SUP` env vars |
| Multi-rail QP setup | ✅ Done | Configurable per-rail QPs (`MPIB_SOUT_QP=2`, `MPIB_SUP_QP=4`) |
| Path isolation (vanilla) | ✅ Done | Island classification; SUP QP suppression for inter-island. See [path_isolation.md](path_isolation.md) |
| Dynamic splitting | ✅ Done | `mpibGetSupBw()` + `mpibComputeSupBytes()`, integer-only |
| SRQ + dynamic rail skipping | ✅ Done | Per-device SRQ; inactive rails skipped entirely; see [dual-rail-cqe.md](dual-rail-cqe.md) |
| GDR (nv_peermem) | ✅ Done | `NCCL_PTR_CUDA` advertised; GPU MR via `ibv_reg_mr` transparent |
| Relaxed ordering | ✅ Done | `ibv_reg_mr_iova2` + `IBV_ACCESS_RELAXED_ORDERING`, default on |
| Flush (`iflush`) | ✅ Done (no-op) | PCIe ordering makes flush unnecessary — see §9 |
| Agent SHM interface | ✅ Done | `mpib_agent_iface.h` seqlock-based hints; agent daemon not yet implemented |

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
* **Connect:** Classifies path (intra/inter-island), creates QPs (suppresses SUP if unreachable).
* **Data Path:** Reads policy via `mpibGetSupBw()` -> Splits data -> Posts to active QPs.
* **Progress:** Polls all CQs -> Aggregates completions.

### B. The Agent (Policy) — *Future*

Intelligent controller (outside plugin scope). Not yet implemented.

* **Role:** Monitor topology, congestion, and link health.
* **Output:** Write per-flow SUP bandwidth hints to SHM.
* **Interface:** Parts-per-1024 encoding via seqlock (no floats).

In vanilla mode (`MPIB_MODE=0`), the agent is not needed — the plugin uses
topology-driven path isolation. In advanced mode (`MPIB_MODE=1`), the plugin
reads agent hints from SHM on every `isend()`. See [path_isolation.md](path_isolation.md) §4.

### C. Agent–Plugin Interface

All definitions live in `include/mpib_agent_iface.h`. The interface has two
parts: a **registration IPC** (Unix domain socket) for connection lifecycle,
and a **hint SHM** (memory-mapped file) for per-transfer policy.

#### C.1 Registration IPC (Unix Socket)

Path: `/tmp/mpib/agent.sock`

When a connection is established (`connect`/`accept`), the plugin sends a
registration request to the agent daemon. The agent assigns a `hint_slot`
index, which the plugin stores in `comm->hint_slot` for the lifetime of the
connection. On `closeSend`/`closeRecv`, the plugin sends a deregistration.

```
Plugin                                  Agent
  │                                       │
  ├── REGISTER ───────────────────────>   │
  │   {conn_id, sout_src_ip,             │
  │    sout_dst_ip, sup_src_ip,           │
  │    sup_dst_ip}                        │
  │                                       │
  │   <──── RESPONSE ─────────────────   │
  │   {status, hint_slot}                 │
  │                                       │
  │   ... data transfers ...              │
  │   (plugin reads SHM[hint_slot]        │
  │    on every isend)                    │
  │                                       │
  ├── DEREGISTER ─────────────────────>   │
  │   {conn_id}                           │
```

The `conn_id` is `(PID << 16 | counter)`, unique per connection per process.
IP addresses are in network byte order. The agent uses the SOUT src/dst pair
to identify the flow and decide what `sup_bw` value to write.

#### C.2 Hint SHM Layout

Path: `/tmp/mpib/hints` (memory-mapped file, 4112 bytes)

```c
struct mpib_hint_shm {
    struct mpib_hint_header header;                       // 16 bytes
    struct mpib_hint_entry entries[MPIB_HINT_MAX_ENTRIES]; // 256 × 16 bytes
};

struct mpib_hint_header {
    uint32_t magic;        // Must be 0x4D504948 ("MPIH")
    uint32_t max_entries;  // 256
};

struct mpib_hint_entry {
    uint32_t sup_bw;  // SUP share, parts-per-1024 [0..1024]
    uint32_t seq;     // Seqlock sequence number
    uint32_t src_ip;  // SOUT source IP (network byte order)
    uint32_t dst_ip;  // SOUT destination IP (network byte order)
};
```

#### C.3 Seqlock Protocol

**Agent writes** (write side of seqlock):
```c
mpib_hint_write(&entries[slot], new_sup_bw);
// Internally: seq++ (odd → write in progress), write sup_bw, seq++ (even → done)
```

**Plugin reads** (read side, called on every `isend` in advanced mode):
```c
uint32_t bw = mpib_hint_read_raw(&entries[slot]);
// Internally: spin while seq is odd; retry if seq changed during read
```

Lock-free, wait-free on the reader side (bounded retries). The agent must not
hold the write lock for extended periods.

#### C.4 How the Plugin Interprets `sup_bw`

The `sup_bw` value from SHM flows through two functions:

**Step 1: `mpibGetSupBw(comm, size)`** — policy entry point (called per `isend`):
```c
if (mode == 0)  // vanilla — never reaches SHM
    return (pathClass == INTRA) ? UINT32_MAX : 0;
if (mode == 1)  // advanced — read agent hint
    return mpib_hint_read_raw(&shm->entries[comm->hint_slot]);
```

**Step 2: `mpibComputeSupBytes(sup_bw, reqSize)`** — converts to byte count:
```c
if (sup_bw == 0)     return 0;                           // 100% SOUT
if (sup_bw >= 1024)  return reqSize;                     // 100% SUP
return (size_t)(((uint64_t)reqSize * sup_bw) >> 10);     // proportional split
```

The result is then 128B-aligned and used to compute `sizeSout` / `sizeSup` per
request. Only rails with non-zero bytes are activated (`active_mask`).

**Agent-facing contract:**

| `sup_bw` written by agent | Plugin behavior |
|--------------------------|-----------------|
| `0` | All data on SOUT. SUP rail idle. |
| `1`–`1023` | Proportional split: SUP gets `reqSize × sup_bw / 1024` bytes (128B aligned). Both rails active. |
| `1024` | All data on SUP. SOUT rail idle (except CTS which always uses SOUT in advanced mode). |
| Not written (default `0`) | Same as `0` — all SOUT. Safe default for unregistered slots. |

**Numeric examples** (for `reqSize = 1 MB = 1048576 bytes`):

| `sup_bw` | SUP bytes | SOUT bytes | `active_mask` |
|----------|-----------|------------|---------------|
| `0` | 0 | 1048576 | `0x1` (SOUT only) |
| `256` | 262144 (25%) | 786432 | `0x3` (both) |
| `512` | 524288 (50%) | 524288 | `0x3` (both) |
| `768` | 786432 (75%) | 262144 | `0x3` (both) |
| `1024` | 1048576 | 0 | `0x2` (SUP only) |

**Important:** The plugin does not clamp or validate `sup_bw` beyond the
`>= 1024` check. Values in `[1025, UINT32_MAX)` are treated as 100% SUP
(same as `1024`). The agent should only write values in `[0, 1024]`.

#### C.5 Timing and Latency

- The plugin reads SHM **once per `isend()` call**, not per byte or per WR.
- The read is a single `uint32_t` load behind a seqlock (typically 1–2 cache
  line reads, < 100 ns).
- The agent can update `sup_bw` at any rate. Changes take effect on the
  **next** `isend()` — there is no batching or buffering.
- If the agent is not running, all SHM entries remain at their initial value
  (`0`), which means 100% SOUT. This is a safe degradation.

#### C.6 Future Extension: BDP Threshold

`mpibGetSupBw()` accepts a `size` parameter (currently unused). A future
enhancement will skip the SHM read for small inter-island transfers:

```c
if (mode == 1 && pc == INTER_ISLAND && size <= BDP_THRESHOLD)
    return 0;  // Small message → SOUT only, skip relay overhead
```

This requires no agent changes — the plugin unilaterally overrides the hint
for small messages.

---

## 3. Detailed Design

### I. Device Management & Implicit Binding

We bypass `makeVDevice`. The binding is implicit: **Logical Device 0 ≡ All Physical Devices**.

* **`init()`**:
  * Reads `MPIB_HCA_SOUT` (scaleout NIC, e.g., `"mlx5_0"`) and `MPIB_HCA_SUP` (scaleup NIC, e.g., `"mlx5_1"`).
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

**`connect(dev=0, handle)` Logic:**

1. Exchange `vProps` (local/remote rail counts) over TCP.
2. **Path classification** from TCP socket addresses (before any QP creation):
   * Extract local SOUT IP from `mpibIfAddr`, remote SOUT IP from `handle->connectAddr`.
   * `mpibIsSameIsland(local, remote)` → `pathClass` (INTRA or INTER).
   * Cache `MPIB_MODE` → `base.mode`.
3. Read `MPIB_SOUT_QP` (default 2) and `MPIB_SUP_QP` (default 4) for per-device QP counts.
4. **SUP QP suppression** (vanilla inter-island only):
   * If `mode == 0 && pathClass == INTER_ISLAND`: set `nqpsSup = 0`.
   * Both sides compute this independently from the same socket IPs, so QP counts agree.
5. `nqps = nqpsSout + nqpsSup` — total QPs across both devices.
6. QP array layout is **contiguous by device**: `[SOUT_0, ..., SOUT_{n-1}, SUP_0, ..., SUP_{m-1}]`
7. For each QP index `q in [0, nqps)`:
   * `devIndex = (q < nqpsSout) ? 0 : 1`
   * Create QP on `mpibDevs[devIndex]`
   * Store `qp->devIndex = devIndex`
8. Exchange metadata (includes `nqpsSout`, `nqpsSup` for validation), call `mpibRtrQp` / `mpibRtsQp`.
9. Map `remDevIdx` for striping remote rkeys.

`mpibAccept` follows the same classification logic, extracting the remote IP from
`rComm->base.sock.addr` (the accepted TCP connection’s peer address).

### III. QP Selection & Striping Strategy

**Overview:**

MPIB uses a two-level data distribution strategy:
1. **Device split**: Each message is divided between SOUT (dev0) and SUP (dev1)
2. **QP round-robin**: Within each device, QPs are selected in round-robin fashion across requests

**Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MPIB_SOUT_QP` | 2 | Number of QPs on SOUT device |
| `MPIB_SUP_QP` | 4 | Number of QPs on SUP device |

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

CTS is pinned to a **single rail** per connection, selected by path isolation policy:

```c
if (mode == 0) {
    // Vanilla: CTS follows strict path isolation
    if (pathClass == INTRA_ISLAND)
        ctsQp = qps[nqpsSout];  // SUP
    else
        ctsQp = qps[0];         // SOUT
} else {
    // Advanced: CTS always on SOUT
    ctsQp = qps[0];
}
```

**Why pin CTS to one rail?** CTS is control-plane traffic (small, frequent).
Pinning to a single QP simplifies signaling: signal every `MPIB_CTS_SIGNAL_INTERVAL`
slots (default 128) to drain the CTS QP without complex per-QP sequence tracking.

**CTS Signaling:**

```c
if ((slot % MPIB_CTS_SIGNAL_INTERVAL) == 0) {
    wr.send_flags |= IBV_SEND_SIGNALED;
}
```

**Suppressed SUP QPs:** When `nqpsSup == 0` (inter-island vanilla), the `qps[]` slots
beyond `nqpsSout` are zero-initialized (`qp == NULL`). The data path guards against
this:

```c
if (qpPtr->qp == NULL) continue;  // SUP suppressed → skip
```

See [path_isolation.md](path_isolation.md) §6 for QP layout details.

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
3. Compute data split:
   * `sup_bw = mpibGetSupBw(comm, size)` — returns parts-per-1024 or sentinel.
   * `sup_bytes = mpibComputeSupBytes(sup_bw, reqSize)` — integer-only, 128B aligned.
   * `sizeSout[r] = reqSize - sup_bytes`, `sizeSup[r] = sup_bytes`.
4. Compute `active_mask` (bit0=SOUT, bit1=SUP) from non-zero sizes.
5. For each device `i in [0, ndevs)`:
   * Call `mpibCommBaseGetQpForRequest(comm, fifoHead, i, &qp, &qpIndex)`
   * If `qp->qp == NULL`: skip (SUP suppressed for inter-island vanilla)
   * If `!(active_mask & devBit)`: skip (device has 0 bytes this transfer)
   * Call `mpibAddEvent(req, qp->devIndex)` to track expected completion
6. Build and post WR chain for each active QP.
7. Increment `fifoHead`, return `*request = req`.

**WR chain structure (per active QP):**

```
wrs[0..nreqs-1]: IBV_WR_RDMA_WRITE (data, not signaled)
wrs[nreqs]:      IBV_WR_RDMA_WRITE_WITH_IMM (signaling WR, signaled)
```

The signaling WR carries `imm_data = (slot | active_mask << 8 | size_q << 10)`. On the leader rail it also writes completion sizes to `remCmplsRecords`; on non-leader rails it is IMM-only (`num_sge=0`).

**Data Split:**

Each request's data is split between SOUT and SUP based on `mpibGetSupBw()`, which
returns a topology-driven constant in vanilla mode or an agent hint in advanced mode.
`mpibComputeSupBytes()` converts the parts-per-1024 value to a byte count using
integer-only arithmetic (no floats). The sender computes `active_mask` (bit0=SOUT,
bit1=SUP) and **only posts to active rails** — inactive rails are skipped entirely.

**Completion model:**

- **SEND:** `events[devIndex]`-based (one event per active QP).
- **RECV:** SRQ + mask-learning. Receiver learns `expected_mask` from the first arriving IMM; completion when `seen_mask == expected_mask`. See [dual-rail-cqe.md](dual-rail-cqe.md).

### V. Progress Engine (`test`)

`mpibTest()` polls both device CQs. Completion criteria differ by request type:

- **SEND:** `events[0] == 0 && events[1] == 0` (same as net_ib).
- **RECV:** `expected_mask != 0 && seen_mask == expected_mask` (SRQ mask-learning). The `expected_mask` is learned from the first arriving IMM's `active_mask` field; each CQE sets a bit in `seen_mask`.

SRQ refill (`mpibSrqCheckAndRefill`) runs in both `irecv` and `test` to prevent RNR.

---

## 4. Implementation History

All phases below are complete. Listed for historical context.

### Phase 1: Single-NIC Baseline ✅

Forked net_ib. Single HCA via `NCCL_IB_HCA`. Validated with `launch_nccl.sh`.

### Phase 2: Dual-NIC Enumeration & Connection ✅

Added `MPIB_HCA_SOUT` / `MPIB_HCA_SUP` env vars. Opens two HCAs, creates merged
vDev with `ndevs=2`. Multi-rail QP creation in `connect`/`accept`.

### Phase 3: Data Splitting & SRQ ✅

Implemented per-request data split between SOUT and SUP. SRQ + mask-learning
protocol for dynamic rail skipping (see [dual-rail-cqe.md](dual-rail-cqe.md)).
Integer-only split via `mpibComputeSupBytes()` — no floats on hot path.

### Phase 4: Path Isolation ✅

Island classification from TCP socket addresses. SUP QP suppression for
inter-island vanilla connections. `mpibGetSupBw()` as the single policy entry
point. SHM hint interface (`mpib_agent_iface.h`) ready for agent. See
[path_isolation.md](path_isolation.md).

### Phase 5: Agent Daemon (Future)

External sidecar daemon that writes per-flow SUP bandwidth hints to SHM.
Requires relay NIC (DOCA Flow VXLAN overlay) for cross-island SUP traffic.
Plugin reads hints via `mpibGetSupBw()` in advanced mode (`MPIB_MODE=1`).
No plugin code changes needed — only `mpibGetSupBw()` already reads SHM when
`mode == 1`.

---

## 5. Testing Matrix

| Test | Command | Expected |
|------|---------|----------|
| Intra-island (vanilla) | `MPIB_MODE=0 ./scripts/launch_nccl.sh` (NP=4, same island) | Pass; 100% SUP; `nqps=6` |
| Inter-island (vanilla) | `MPIB_MODE=0 ./scripts/launch_nccl.sh` (NP=8, cross-island) | Pass; 100% SOUT for cross-island conns; `nqps=2` |
| Mixed (vanilla) | `MPIB_MODE=0 ./scripts/launch_nccl.sh` (NP=8, tree topology) | Pass; intra=SUP, inter=SOUT |
| Advanced mode | `MPIB_MODE=1 ./scripts/launch_nccl.sh` + agent | (Future) Agent-driven split |

---

## 6. Risk & Mitigation

| Risk | Mitigation |
|------|------------|
| Cross-QP ordering with split | Per-rail IMM (net_ib-style), one CQE per rail |
| Uneven MTU across NICs | Take `min(mtu)` in connect handshake |
| MR registration per-device | `mpibMrHandle.mrs[]` already per-rail |
| SUP QP access when suppressed | `qpPtr->qp == NULL` guard in data path |
| QP count mismatch (sender/receiver) | Both sides classify independently from same socket IPs |

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
