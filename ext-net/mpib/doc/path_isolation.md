# MPIB Path Isolation Design

## 1. Overview

MPIB is a dual-rail NCCL net plugin. Each node exposes two physical HCAs as a
single NCCL virtual device:

| Rail | HCA | Role | Network |
|------|-----|------|---------|
| **SOUT** (dev 0) | `mlx5_0` | Primary data path, routable across islands | ToR-switched fabric (e.g. `10.0.1.0/24`, `10.0.2.0/24`) |
| **SUP** (dev 1) | `mlx5_1` | Supplementary data path, intra-island only | Direct-attach fabric (e.g. `10.9.x.x/24`) |

The SUP fabric provides extra bandwidth between nodes within the same island but has no physical route to nodes in other islands. The SOUT fabric is fully routable. Path isolation ensures that MPIB never attempts to send SUP traffic to an unreachable destination.

**Terminology.** An *island* is a set of nodes sharing a common subnet.
Two nodes are *intra-island* if their SOUT IPs share the same prefix (default
`/24`); otherwise they are *inter-island*.

---

## 2. Architecture: Classification vs. Policy

The design separates two concerns:

| Concern | Granularity | When | Stored where |
|---------|------------|------|--------------|
| **Classification** — *can* SUP reach the peer? | Connection (once) | `connect()` / `accept()` | `base.pathClass`, `base.mode` |
| **Policy** — *how much* traffic goes to SUP? | Transfer (every `isend()`) | `mpibGetSupBw()` | Return value, not stored |

Classification is a physical property of the topology and does not change for
the lifetime of a connection. Policy is a runtime decision that can vary per
transfer, enabling future agent-driven control without changing the
classification layer.

### 2.1 Connection-level fields

```c
typedef enum {
  MPIB_PATH_INTRA_ISLAND = 0,   // Same SOUT subnet
  MPIB_PATH_INTER_ISLAND = 1,   // Different SOUT subnet
} mpibPathClass;

struct mpibNetCommBase {
  ...
  mpibPathClass pathClass;  // Computed once from socket IPs
  int32_t       mode;       // Cached from MPIB_MODE env (0=vanilla, 1=advanced)
  uint32_t      nqps;       // Total QP count (nqpsSout + nqpsSup)
  uint32_t      nqpsSout;   // QPs on SOUT
  uint32_t      nqpsSup;    // QPs on SUP (may be 0 if suppressed)
  ...
};
```

### 2.2 Transfer-level decision

```c
uint32_t mpibGetSupBw(struct mpibSendComm *comm, size_t size);
```

Returns a *parts-per-1024* value (or sentinel) indicating how much of the
transfer should use SUP. This is the **sole policy entry point**; `mpibIsend()`
does not read `pathClass` or `mode` directly.

---

## 3. Vanilla Mode (`MPIB_MODE=0`)

Vanilla mode implements strict, topology-driven path isolation with zero
runtime overhead. No relay is required. No shared memory is read.

### 3.1 Connection setup

Classification happens **before** any QP is created, using the TCP control
socket addresses that are already available at that point:

```
mpibConnect / mpibAccept
  │
  ├─ Extract local IP from mpibIfAddr (SOUT interface)
  ├─ Extract remote IP from TCP socket peer address
  ├─ mpibIsSameIsland(local, remote)  → pathClass
  ├─ Cache MPIB_MODE → base.mode
  │
  ├─ Set nqpsSout, nqpsSup from env params
  ├─ if (mode==0 && pathClass==INTER_ISLAND):
  │      nqpsSup = 0          ← suppress SUP QPs entirely
  │
  ├─ nqps = nqpsSout + nqpsSup
  └─ Create only nqps QPs (contiguous: [SOUT₀..SOUTₙ₋₁, SUP₀..SUPₘ₋₁])
```

Both sender (`mpibConnect`) and receiver (`mpibAccept`) independently compute
the same classification from the same socket IPs, so their QP counts always
agree without an extra protocol exchange. Note that the fundamental **assumption** here is that the TCP socket uses SOUT NIC, which is always true in my testbed.

**Key design decision: early QP suppression.** For inter-island vanilla
connections, SUP QPs are never created. This avoids `ibv_modify_qp` timeouts
(the QP RTR transition would fail because the SUP fabric has no route to the
remote peer). The `qps[]` array slots beyond `nqps` remain zero-initialized
(`qp == NULL`), which might seem dangerous but lets keep it that way.

### 3.2 Data path

`mpibGetSupBw()` in vanilla mode is trivial:

```c
if (mode == 0) {
  return (pathClass == INTRA_ISLAND) ? UINT32_MAX   // 100% SUP
                                     : 0;           // 100% SOUT
}
```

This drives the data split computation:

```c
static inline size_t mpibComputeSupBytes(uint32_t sup_bw, size_t reqSize) {
  if (sup_bw == 0)     return 0;          // All SOUT
  if (sup_bw >= 1024)  return reqSize;    // All SUP (includes UINT32_MAX)
  return (size_t)(((uint64_t)reqSize * sup_bw) >> 10);
}
```

The resulting `active_mask` (bit 0 = SOUT, bit 1 = SUP) determines which
rails post work requests. The QP iteration loop includes a guard for
suppressed devices:

```c
const int nqps = mpibCommBaseGetNqpsPerRequest(&comm->base);  // = ndevs = 2
for (int i = 0; i < nqps; i++) {
  mpibQp *qpPtr;
  NCCLCHECK(mpibCommBaseGetQpForRequest(&comm->base, fifoHead, i, &qpPtr, &qpIdx));
  if (qpPtr->qp == NULL) continue;   // SUP suppressed → skip
  ...
}
```

### 3.3 CTS routing

CTS RDMA writes follow strict path isolation:

| pathClass | CTS rail | QP used |
|-----------|----------|---------|
| `INTRA_ISLAND` | SUP | `qps[nqpsSout]` |
| `INTER_ISLAND` | SOUT | `qps[0]` |

### 3.4 Vanilla behavior summary

| Connection type | QPs created | Data rail | CTS rail | SHM reads |
|----------------|-------------|-----------|----------|-----------|
| Intra-island | `nqpsSout + nqpsSup` (e.g. 2+4=6) | SUP only | SUP | 0 |
| Inter-island | `nqpsSout` only (e.g. 2) | SOUT only | SOUT | 0 |

---

## 4. Advanced Mode (`MPIB_MODE=1`) — Future

Advanced mode enables an external agent process to control the SOUT/SUP split
ratio per-transfer via shared memory hints. The classification layer is
unchanged; only the policy function differs.

### 4.1 What changes

- `mpibGetSupBw()` reads the agent hint from SHM instead of returning a
  fixed value.
- SUP QPs are **always created**, even for inter-island connections, because
  the relay NIC (DOCA Flow VXLAN overlay) bridges the SUP fabric across
  islands.
- CTS always routes on SOUT (the agent controls data traffic only, not the
  control path).

### 4.2 `mpibGetSupBw()` in advanced mode

```c
if (mode == 1) {
  // Future: gate on `size > BDP_threshold` for inter-island
  return mpibReadHintRaw(comm->hint_slot);  // seqlock read from SHM
}
```

The SHM `sup_bw` field uses parts-per-1024 encoding:

| Value | Meaning |
|-------|---------|
| `0` | 100% SOUT |
| `512` | 50/50 split |
| `1024` | 100% SUP |

### 4.3 SUP reachability guard

The single guard that ties classification to QP creation is:

```c
bool suppress_sup = (mode == 0 && pathClass == INTER_ISLAND);
```

In advanced mode this is always `false`, because the relay NIC makes SUP
cross-island reachable. This is the only line in the codebase that couples
mode to QP creation policy.

### 4.4 Future extension points

All live in `mpibGetSupBw()`:

- **BDP threshold**: skip agent read when `size ≤ BDP` for inter-island
  connections (small messages go SOUT-only to avoid relay overhead).
- **Per-connection agent hints**: different hint slots per connection allow
  flow-level differentiation.
- **Fallback on agent failure**: if SHM is stale, fall back to vanilla
  policy.

No structural changes are needed for any of these — they are conditionals
inside `mpibGetSupBw()`.

---

## 5. Island Classification

Two nodes are in the same island if their SOUT IPs share the same `/24`
prefix (configurable via `MPIB_ISLAND_PREFIX_LEN`).

```c
static int mpibIsSameIsland(uint32_t ip_a, uint32_t ip_b) {
  int prefix = mpibParamIslandPrefixLen();   // default 24
  uint32_t mask = (prefix == 0) ? 0 : (~0u << (32 - prefix));
  return (ip_a & mask) == (ip_b & mask);
}
```

The SOUT IPs are obtained from the TCP control socket addresses via `mpibSocketAddrToIpv4()`.

---

## 6. QP Layout

QPs are stored contiguously in `base.qps[]`:

```
Index:  0        1        ...  nqpsSout-1   nqpsSout  ...  nqps-1
Rail:   SOUT     SOUT          SOUT         SUP            SUP
```

When `nqpsSup == 0` (inter-island vanilla), only indices `[0, nqpsSout)` hold
valid QP objects. Indices `≥ nqpsSout` have `qp == NULL` (zero-initialized
struct).

`mpibCommBaseGetQpForRequest(base, id, devIndex, &qp, &qpIdx)` selects a QP
by device index using round-robin within that device's pool:

- `devIndex == 0` → `qpIdx = id % nqpsSout`
- `devIndex == 1` → `qpIdx = nqpsSout + (id % nqpsSup)`

When `nqpsSup == 0`, the function returns a pointer to the zero-initialized
struct at `qps[nqpsSout]`. Callers check `qpPtr->qp == NULL` to detect this
and skip the rail.

---

## 7. Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `MPIB_MODE` | `0` | `0` = vanilla (strict isolation, no agent), `1` = advanced (agent-driven) |
| `MPIB_ISLAND_PREFIX_LEN` | `24` | SOUT IP prefix length for island classification |
| `MPIB_SOUT_QP` | `2` | Number of QPs on SOUT rail |
| `MPIB_SUP_QP` | `4` | Number of QPs on SUP rail (set to 0 at runtime for inter-island vanilla) |
| `MPIB_HCA_SOUT` | *(required)* | SOUT HCA name (e.g. `mlx5_0`) |
| `MPIB_HCA_SUP` | *(required)* | SUP HCA name (e.g. `mlx5_1`) |
| `MPIB_OOB_IF` | *(required)* | OOB interface for TCP control |

---

## 8. Source File Map

| File | Relevant functions | Role |
|------|-------------------|------|
| `mpib_common.h` | `mpibPathClass`, `mpibNetCommBase`, `mpibCommBaseGetQpForRequest` | Struct definitions, QP selection |
| `mpib_connect.cc` | `mpibConnect`, `mpibAccept`, `mpibSocketAddrToIpv4`, `mpibIsSameIsland` | Classification, QP creation/suppression |
| `mpib_p2p.cc` | `mpibIsend`, `mpibComputeSupBytes` | Data split, WR posting, `qp==NULL` guard |
| `mpib_p2p.h` | `mpibRecvCommGetQpForCts` | CTS rail selection |
| `mpib_agent_client.cc` | `mpibGetSupBw` | Policy entry point (vanilla fast-path / advanced SHM read) |
| `mpib_agent_iface.h` | `mpib_hint_entry`, `mpib_hint_read_raw` | SHM interface for agent hints |
