# Baseline: Isolated Path Separation (SUP / SOUT)

## 1. Problem Statement

MPIB emulates NVLink-like scaleup connectivity using a second RDMA NIC (SUP).
In a real NVLink system, NCCL discovers the NVLink topology and confines
intra-node traffic to NVLink automatically. Since MPIB's "scaleup" is just
another RDMA NIC, NCCL's topology discovery cannot distinguish it from a
regular network device. Without intervention, NCCL will treat both NICs
identically and the plugin's data split (currently agent-driven) may route
traffic on the wrong rail.

**Goal:** For a fair vanilla single-path baseline, enforce strict path
separation:

| Traffic class | Path | Rationale |
|---------------|------|-----------|
| Intra-island (same scaleup domain) | SUP only | Emulates NVLink—low-latency, high-BW |
| Inter-island (cross scaleup domain) | SOUT only | Scaleout fabric; must traverse ToR/spine |

No multi-path splitting. Every connection uses exactly one of the two NICs.

---

## 2. Background

### 2.1 Topology (from `vm-topology.yaml`)

```
Island A: vm1, vm2, vm3, vm4       Island B: vm5, vm6, vm7, vm8
  dev0 (SOUT): 10.0.1.{1-4}/24       dev0 (SOUT): 10.0.2.{5-8}/24
  dev1 (SUP):  10.9.1.{1-4}/24       dev1 (SUP):  10.9.2.{5-8}/24
```

**Key observation:** The SOUT IP subnet encodes island membership.
`10.0.1.0/24` = Island A, `10.0.2.0/24` = Island B. A /24 prefix match
on SOUT IPs is necessary and sufficient to determine same-island membership.

### 2.2 Current Data Path

1. Plugin presents **one logical device** (`ndev=1`) to NCCL.
2. Internally manages two physical NICs: dev\[0\]=SOUT, dev\[1\]=SUP.
3. On every `isend()`, reads `sup_bw` from agent hint SHM (seqlock).
4. Computes `sup_ratio = sup_bw / (1 + sup_bw)`, splits data between rails.
5. Builds `active_mask` (bit0=SOUT, bit1=SUP), posts WRs only to active rails.
6. Inactive rails are **completely skipped** (no zero-byte posts).

The mechanism for single-rail operation already exists—`active_mask = 0x1`
means SOUT-only, `active_mask = 0x2` means SUP-only. What's missing is
the **policy** to set the right mask per connection.

### 2.3 Transport Agent

- Plugin **requires** the agent at init (`mpibAgentClientInit()` fails →
  plugin rejected by NCCL). 
- Agent provides hint SHM (`/tmp/mpib/hints`) and registration socket
  (`/tmp/mpib/agent.sock`).
- The plugin classifies each connection as SUP-eligible or SOUT-only at
  setup time. SUP-eligible connections still consult the agent's `sup_bw`
  hint on every `isend()`, so the agent retains runtime control over SUP
  traffic. SOUT-only connections bypass the hint and use a cached value.

---

## 3. Approach: Plugin-Side Subnet Classification

Classify connections at `connect`/`accept` time inside the plugin, using a
SOUT IP subnet comparison. The transport agent remains a required dependency
(provides SHM and registration infrastructure), but the baseline plugin does
not consult agent hints at runtime—it uses its own cached classification.

| Property | Value |
|----------|-------|
| Classification point | `connect`/`accept` (one-time, at setup) |
| Classification input | SOUT src/dst IPv4 from exchanged GIDs |
| Per-send overhead | Zero for SOUT (cached); one seqlock read for SUP (existing cost) |
| Agent dependency | Required (SHM + IPC); hints consulted at runtime for SUP connections |
| Code changes | ~50 lines across 4 files |
| Migration to multi-path | Set `hintActive=1` for all connections (or change classification rule) |

---

## 4. Detailed Design

### 4.1 Classification Rule

At connection establishment, the plugin has both local and remote GIDs for
each device (exchanged in `mpibConnectionMetadata.devs[]`). It extracts IPv4
addresses from the SOUT GIDs and applies a subnet mask comparison:

```
sout_src_ip = mpibGidToIpv4(local SOUT GID)    // dev[0] local
sout_dst_ip = mpibGidToIpv4(remote SOUT GID)   // dev[0] remote
mask        = 0xFFFFFF00                        // /24 default

same_island = (sout_src_ip & mask) == (sout_dst_ip & mask)
```

Result:

| Condition | `hintActive` | `pathSupBw` | Runtime `sup_bw` source |
|-----------|-------------|-------------|-------------------------|
| Same island | `1` | `UINT32_MAX` (fallback) | `mpibAgentReadHint()` → agent controls SUP share |
| Different island | `0` | `0` | `pathSupBw` → SOUT only |

**Configurable mask:** Environment variable `MPIB_ISLAND_PREFIX_LEN` (default
`24`) specifies the prefix length. This accommodates future topologies where
island boundaries fall on different subnet boundaries.

### 4.2 Data Structures

Add to `mpibNetCommBase` (shared by send and recv comms):

```c
// Plugin-side path classification (computed once at connect/accept)
uint32_t pathSupBw;    // Fallback sup_bw: 0=SOUT-only, UINT32_MAX=SUP-only
int      hintActive;   // 1 = consult agent hint at runtime, 0 = use pathSupBw
```

### 4.3 Connection Flow Changes

#### `mpibConnect()` / `mpibAccept()` — after GID exchange

```
1. Extract SOUT src/dst IPv4 from GIDs (existing code)
2. Compute island membership:
     mask = inet_prefix_mask(MPIB_ISLAND_PREFIX_LEN)
     same_island = (sout_src & mask) == (sout_dst & mask)
     comm->base.pathSupBw  = same_island ? UINT32_MAX : 0
     comm->base.hintActive = same_island ? 1 : 0
3. Agent registration (existing code, unchanged — still fatal on failure):
     mpibAgentRegister(...) → hint_slot
4. Log classification:
     INFO("NET/MPIB : path=%s (sout_src=%s, sout_dst=%s)",
          same_island ? "SUP" : "SOUT", ...)
```

### 4.4 Send Path Changes

In `mpibIsend()`, gate the hint read on `hintActive`:

```c
// BEFORE:
const uint32_t sup_bw = mpibAgentReadHint(comm->hint_slot);

// AFTER:
const uint32_t sup_bw = comm->base.hintActive
    ? mpibAgentReadHint(comm->hint_slot)
    : comm->base.pathSupBw;
```

- **SOUT-only** (`hintActive=0`): `pathSupBw=0` → `sup_ratio=0.0` →
  `active_mask=0x1`. Zero per-send overhead (no SHM read).
- **SUP-eligible** (`hintActive=1`): agent hint consulted every `isend()`.
  Dummy agent returns large `sup_bw` → `active_mask=0x2` (SUP only).
  Future agents can return any value to enable dynamic control.

Everything downstream (ratio computation, `active_mask`, WR posting) is
unchanged.

### 4.5 CTS Path Changes

Currently, CTS is **pinned to SOUT** (`mpibRecvCommGetQpForCts` always returns
`qps[0]`). For SUP-only connections, CTS must go on the SUP rail:

```c
// BEFORE (mpib_p2p.h):
static inline ncclResult_t
mpibRecvCommGetQpForCts(struct mpibRecvComm *recvComm, uint32_t id,
                        mpibQp **qp) {
  (void)id;
  *qp = &recvComm->base.qps[0];  // Always SOUT
  return ncclSuccess;
}

// AFTER:
static inline ncclResult_t
mpibRecvCommGetQpForCts(struct mpibRecvComm *recvComm, uint32_t id,
                        mpibQp **qp) {
  (void)id;
  // Use SUP QP for CTS when connection is SUP-only
  if (recvComm->base.pathSupBw == UINT32_MAX) {
    // SUP-only: use first SUP QP
    *qp = &recvComm->base.qps[recvComm->base.nqpsSout];
  } else {
    // SOUT-only or mixed: use first SOUT QP
    *qp = &recvComm->base.qps[0];
  }
  return ncclSuccess;
}
```

This ensures **all** RDMA traffic (data + CTS) stays on the designated rail.

### 4.6 Correctness Argument

| Property | Why it holds |
|----------|-------------|
| **Data isolation** | `active_mask` is `0x1` or `0x2`; inactive rail is never posted to |
| **CTS isolation** | CTS QP selection matches data rail (§4.5) |
| **Completion correctness** | SEND: `events[devIndex]` only set for active rail; RECV: `expected_mask` learned from sender's IMM matches single-rail mask |
| **SRQ safety** | SRQ refill runs on both devices regardless (unused SRQ has no harm); active device's SRQ receives IMM completions normally |
| **No stale CTS** | CTS carries remote rkeys for both rails; sender uses the active rail's rkey for data |
| **Agent coexistence** | SUP connections already consult agent hints; agent can dynamically adjust SUP traffic without plugin changes. SOUT connections bypass hints (no SHM read). Extending `hintActive=1` to all connections enables full agent-driven multi-path |

---

## 5. Implementation Plan

### Step 1: Add `pathSupBw` and `hintActive` to `mpibNetCommBase`

**File:** `src/mpib_common.h`

Add two fields to `struct mpibNetCommBase`:
```c
uint32_t pathSupBw;   // Fallback sup_bw (0 or UINT32_MAX)
int      hintActive;  // 1 = read agent hint at runtime, 0 = use pathSupBw
```

Initialize both to `0` in existing comm allocation code.

### Step 2: Add `MPIB_ISLAND_PREFIX_LEN` param

**File:** `src/mpib_init.cc` (or `src/mpib_param.cc`)

```c
MPIB_PARAM(IslandPrefixLen, "ISLAND_PREFIX_LEN", 24);
```

### Step 3: Add island classification helper

**File:** `src/mpib_connect.cc`

```c
// Returns 1 if src and dst are in the same island (same SOUT subnet)
static int mpibIsSameIsland(uint32_t sout_src_ip, uint32_t sout_dst_ip) {
  int prefixLen = (int)mpibParamIslandPrefixLen();
  if (prefixLen <= 0 || prefixLen > 32) prefixLen = 24;
  uint32_t mask = (prefixLen == 32) ? 0xFFFFFFFF : ~((1u << (32 - prefixLen)) - 1);
  return (sout_src_ip & mask) == (sout_dst_ip & mask);
}
```

### Step 4: Integrate classification in `mpibConnect()` and `mpibAccept()`

**File:** `src/mpib_connect.cc`

After the existing GID → IPv4 extraction block (both connect and accept),
add island classification before agent registration:

```c
// Classify connection path
int sameIsland = mpibIsSameIsland(sout_src_ip, sout_dst_ip);
comm->base.pathSupBw  = sameIsland ? UINT32_MAX : 0;
comm->base.hintActive = sameIsland ? 1 : 0;
INFO(NCCL_NET, "NET/MPIB : Path classification: %s hintActive=%d "
     "(sout_src=0x%08x sout_dst=0x%08x)",
     sameIsland ? "SUP (intra-island)" : "SOUT (inter-island)",
     comm->base.hintActive, sout_src_ip, sout_dst_ip);
```

Agent registration remains unchanged (still fatal on failure).

Apply the **same pattern** in both `mpibConnect()` and `mpibAccept()`.

### Step 5: Update `mpibIsend()` hint source

**File:** `src/mpib_p2p.cc`

Replace:
```c
const uint32_t sup_bw = mpibAgentReadHint(comm->hint_slot);
```
With:
```c
const uint32_t sup_bw = comm->base.hintActive
    ? mpibAgentReadHint(comm->hint_slot)
    : comm->base.pathSupBw;
```

### Step 6: Update CTS QP selection

**File:** `src/mpib_p2p.h`

Modify `mpibRecvCommGetQpForCts()` to route CTS on the correct rail based
on `pathSupBw` (see §4.5).

---

## 6. Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `MPIB_ISLAND_PREFIX_LEN` | `24` | SOUT IP prefix length for island comparison. Same /N prefix → same island → SUP. |
| `MPIB_HCA_SOUT` | *(required)* | SOUT HCA name (e.g., `mlx5_0`) |
| `MPIB_HCA_SUP` | *(required)* | SUP HCA name (e.g., `mlx5_1`) |
| `MPIB_OOB_IF` | *(required)* | OOB network interface for TCP control |

**Agent required.** A standalone dummy agent providing `/tmp/mpib/hints` and
`/tmp/mpib/agent.sock` must be running before NCCL starts. SUP-eligible
connections (`hintActive=1`) read the agent's `sup_bw` hint on every
`isend()`. The dummy agent should set `sup_bw` to a large value (e.g.,
`UINT32_MAX`) for SUP-only baseline behavior.

### Example: Baseline launch

```bash
# Standalone dummy agent must be running (provides /tmp/mpib/hints + agent.sock)
export MPIB_HCA_SOUT=mlx5_0
export MPIB_HCA_SUP=mlx5_1
export MPIB_OOB_IF=eth0
export MPIB_ISLAND_PREFIX_LEN=24

export NCCL_NET_PLUGIN=mpib
# NCCL allreduce will use:
#   - SUP for vm1↔vm2 (10.0.1.1 ↔ 10.0.1.2, same /24)
#   - SOUT for vm1↔vm5 (10.0.1.1 ↔ 10.0.2.5, different /24)
mpirun -np 8 ... nccl-tests/build/all_reduce_perf
```

---

## 7. Verification & Testing

### 7.1 Unit Validation

**Subnet classification logic:**
```
mpibIsSameIsland(10.0.1.1, 10.0.1.4)  → 1 (same island)
mpibIsSameIsland(10.0.1.1, 10.0.2.5)  → 0 (different island)
mpibIsSameIsland(10.0.1.1, 10.0.1.1)  → 1 (loopback, same island)
```

### 7.2 Smoke Test: Plugin Load + Path Assignment

Run a 2-node NCCL allreduce with `NCCL_DEBUG=INFO`:

```bash
# Same island (vm1 ↔ vm2): expect SUP path
NCCL_DEBUG=INFO NCCL_NET_PLUGIN=mpib mpirun -np 2 --host vm1,vm2 \
    nccl-tests/build/all_reduce_perf -b 8 -e 128M

# Verify in logs:
#   NET/MPIB : Path classification: SUP (intra-island) (sout_src=... sout_dst=...)
#   No SOUT-related ibv_post_send calls in trace
```

```bash
# Different island (vm1 ↔ vm5): expect SOUT path
NCCL_DEBUG=INFO NCCL_NET_PLUGIN=mpib mpirun -np 2 --host vm1,vm5 \
    nccl-tests/build/all_reduce_perf -b 8 -e 128M

# Verify in logs:
#   NET/MPIB : Path classification: SOUT (inter-island) (sout_src=... sout_dst=...)
```

### 7.3 Functional Test: Mixed Traffic (4-node, 2 per island)

```bash
# vm1, vm2 (Island A) + vm5, vm6 (Island B)
# Ring: vm1→vm2→vm5→vm6→vm1
#   vm1↔vm2: SUP (intra-island)
#   vm2↔vm5: SOUT (inter-island)
#   vm5↔vm6: SUP (intra-island)
#   vm6↔vm1: SOUT (inter-island)
NCCL_DEBUG=INFO NCCL_NET_PLUGIN=mpib mpirun -np 4 --host vm1,vm2,vm5,vm6 \
    nccl-tests/build/all_reduce_perf -b 8 -e 1G
```

### 7.4 Wire Verification (optional)

Use `ibv_devinfo` + `perfquery` or `rdma_bw` on each NIC to confirm:
- SUP NIC shows traffic only for intra-island pairs
- SOUT NIC shows traffic only for inter-island pairs

---

## 8. Future Extensions

1. **CTS signaling interval tuning:** The current `MPIB_CTS_SIGNAL_INTERVAL`
   (128) was tuned assuming all CTS flows through the SOUT QP. With path
   separation, SUP-only connections route CTS through a SUP QP instead.
   If the SUP CTS QP's send queue fills up under heavy load, the interval
   may need per-QP adjustment. Not a correctness issue — monitor for
   `IBV_WC_RETRY_EXC_ERR` or send queue full errors during stress tests.


2. **Full agent-driven multi-path:** Set `hintActive=1` by varying classification rule. The agent can then set `sup_bw`
   to any value per connection, enabling dynamic multi-path splitting.
   No `isend()` code changes needed—the conditional hint read already
   supports this.

