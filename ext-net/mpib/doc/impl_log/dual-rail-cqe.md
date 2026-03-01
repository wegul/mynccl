# Dual-rail CQE: Eliminating 0-byte IMMs

This document describes potential optimizations to eliminate 0-byte `RDMA_WRITE_WITH_IMM` completions on inactive rails.

---

## Resolved: CTS Signaling Design

### Problem

The original CTS signaling logic from net_ib:

```c
if (slot == ctsQp->devIndex) {
  wr.send_flags |= IBV_SEND_SIGNALED;
}
```

This works when CTS uses **one fixed QP per device**. Early mpib iterations tried to spread CTS across multiple QPs per device (round-robin), which broke the signaling correlation and caused send queue overflow.

### Solution: Match net_ib's Design

**CTS uses one fixed QP per device** (the first QP on each device):
- Device 0 (SOUT): `qps[0]`
- Device 1 (SUP): `qps[nqpsSout]`

With this design, the simple signaling rule `slot == devIndex` works correctly:
- Device 0's CTS QP signals at slots 0, 2, 4, ...
- Device 1's CTS QP signals at slots 1, 3, 5, ...

**Data QPs** (for `isend`/`irecv` payload) still use per-device round-robin to utilize all QPs for bandwidth. Only CTS is pinned.

Note: MPIB does **not** stripe a single request across multiple QPs on the same device. “Round-robin” here means selecting **one QP per device per request**.

**Why this separation?**
- CTS is control-plane (small messages, must never stall)
- Data is bandwidth-plane (benefits from multi-QP parallelism)
- Pinning CTS to one QP per device keeps signaling simple and robust

---

## Current behavior

MPIB uses `RDMA_WRITE_WITH_IMM` for receiver-side completion:

- Sender posts `IBV_WR_RDMA_WRITE_WITH_IMM` on each participating QP (one per device).
- Receiver posts `ibv_post_recv()` on each QP to catch the IMM and generate `IBV_WC_RECV_RDMA_WITH_IMM`.
- `mpibTest()` tracks expected CQEs via `req->events[devIndex]`.

**Key constraint**: Every `RDMA_WRITE_WITH_IMM` requires a posted RECV WQE on that same remote QP.

When a device carries 0 bytes (e.g., 100% SUP split), the current code still posts a 0-byte IMM as a "completion doorbell". This works but is wasteful:

- Extra WR posting overhead on sender.
- Extra RECV WQE consumed on receiver.
- Extra CQE polling in `mpibTest()`.

---

## Proposed Design: SRQ-Based Dynamic Rail Splitting

To eliminate the overhead of 0-byte messages on inactive rails—while maintaining the sender's ability to make last-microsecond routing decisions—we propose shifting from per-QP Receive Queues (RQ) to a per-device Shared Receive Queue (SRQ) architecture.

### 1. Functional Requirements
*   **Zero Overhead on Inactive Rails:** If the sender chooses a 100/0 split, the inactive rail MUST NOT send any packets (no 0-byte IMM).
*   **Fresh Telemetry:** The routing decision (active rail mask) MUST be made by the sender inside `mpibIsend`, immediately before posting work requests.
*   **Robustness:** The receiver MUST NOT become desynchronized (e.g., "Ghost WQE" drift) if the sender skips a rail.

### 2. Technical Architecture

#### 2.1 Resource Model (Receiver)
*   **SRQ per Device:** Create one `ibv_srq` per IB device context (`ibv_context`). All QPs belonging to that device share this SRQ.
*   **Generic WQEs:** The receiver posts "generic" Receive WRs to the SRQ. These WQEs are not bound to specific requests or slots.
*   **Refill Policy:**
    *   **Primary:** Check/refill SRQ depth at the start of `mpibIrecv`.
    *   **Safety:** Check/refill in `mpibTest` to ensure progress and prevent RNR scenarios during long compute phases.

#### 2.2 Protocol: Metadata IMM & Size Side-Channel
Since the SRQ WQE is generic, `wc->wr_id` is useless for identifying the request. We must embed routing metadata in `imm_data`.

**New `imm_data` Layout (32 bits):**

MPIB has exactly **2 rails**, so only **2 bits** are needed for the active mask.

```c
// Conceptual layout. Use explicit bit packing/unpacking (shifts/masks) in code;
// do not rely on C bitfield ABI.
//
//  [ 7:0]   slot_idx      (0..255)
//  [ 9:8]   active_mask   (bit0=SOUT, bit1=SUP; values 1,2,3)
//  [31:10]  size_q        (quantized total size for nreqs==1, or sentinel)
```

**Risk note:** With SRQ, the receiver identifies a request from `imm_data.slot_idx` (slot-only). This assumes the sender never produces a late/duplicate IMM for a slot after that slot has been reused. Keep a short code comment + lightweight debug checks during bring-up.

**Using `size_q` for size**

The `size_q` field can encode the (total) bytes sent for the request with a fixed granularity.

Define:

- `MPIB_IMM_SIZE_G = 128` bytes (granularity, must match your alignment assumptions)
- `MPIB_IMM_SIZEQ_BITS = 22`
- `MPIB_IMM_SIZEQ_SENTINEL = (1u << MPIB_IMM_SIZEQ_BITS) - 1`  // all ones

- `size_q = ceil_div(size_bytes, MPIB_IMM_SIZE_G)`
- Receiver reconstructs `size_bytes = size_q * MPIB_IMM_SIZE_G` and clamps to the posted receive size.

This is only valid if the maximum representable size is sufficient.

Because `size_q` is 22 bits in this layout, the max is much larger than a 16-bit scheme:

- With `MPIB_IMM_SIZE_G=128B`, max encodable size is $(2^{22}-1)\times128 \approx 512\text{MiB}$

### Corner case: `nreqs==1` messages > max encodable

In the common case (`nreqs==1` and size ≤ max encodable), the sender can encode the real sent size in `size_q` and avoid touching `cmplsRecords`.

However, a rare but real corner case is `nreqs==1` with size > max encodable (e.g., >~512MiB when `MPIB_IMM_SIZE_G=128B`). In that case the sender must fall back to the completion-record side channel.

**Sentinel approach:** reserve the maximum `size_q` value as a sentinel:

- If `size_q == MPIB_IMM_SIZEQ_SENTINEL`, the receiver interprets this as “size is not encoded; read exact size from `cmplsRecords[slot][0]`”.

**WR count implication (for 2 rails):**

- Common path: `1 request → N active rails → N payload WRs` (each active rail posts one `RDMA_WRITE_WITH_IMM` carrying payload + metadata IMM)
- `nreqs>1` path:
    - Sharding is still computed **per request** (each of the `nreqs` buffers gets its own split between SOUT/SUP).
    - The completion `active_mask` is a **per-slot** property: `active_mask = OR(mask_i for i in 0..nreqs-1)`.
    - Posting shape: `1 slot (nreqs>1) → N active rails → N * (nreqs payload WRs + 1 signaling WR)`
        - Each active rail posts `nreqs` payload `RDMA_WRITE` WRs (some per-request WRs may be 0-length on that rail if that specific request’s split gives it 0 bytes).
        - Each active rail posts exactly one signaling `RDMA_WRITE_WITH_IMM` WR to satisfy mask-based completion.
        - Only the leader rail attaches an SGE to the signaling WR to write `cmplsRecords[slot][0..nreqs-1]`.
        - Non-leader active rails send the signaling WR with `num_sge=0` (IMM-only).
- Overflow path (`size_q==MPIB_IMM_SIZEQ_SENTINEL`): `1 request → N active rails → N payload WRs + 1 extra WR` where the extra WR (leader rail only) writes `cmplsRecords[slot][0]` (exact byte size) into receiver memory.

This keeps the “easy start” simple and pays extra complexity only for the >max corner case.

**Size Delivery:**

MPIB already has a “completion-record side channel” (`remCmplsRecords` → receiver `cmplsRecords`). For `nreqs > 1`, the sizes are written to receiver memory by the **same signaling WR** (`IBV_WR_RDMA_WRITE_WITH_IMM`) by attaching an SGE which targets `remCmplsRecords[slot]`.

With SRQ + metadata IMM, `imm_data` is repurposed for protocol fields `(slot, mask, size_q)`, so the receiver can no longer rely on `imm_data` to carry the payload size in the current (single-size) form. Therefore:

- The sender should deliver sizes via `remCmplsRecords` (for `nreqs >= 1`), and the receiver should read sizes from `cmplsRecords[slot]`.
- Only **one rail** should perform the size-record write (“leader rail”) to avoid redundant PCIe/NIC work.

**Leader rail selection (deterministic):** The sender chooses the leader rail as a pure function of `active_mask`:

- If `active_mask & 0x1` (SOUT active) → leader = SOUT
- Else (only bit1 set) → leader = SUP

The receiver does not need to know which rail was leader; it only needs the fact that `cmplsRecords[slot]` is populated when required.

Note: If `size_q` encodes size for `nreqs==1`, the implementation may skip the `cmplsRecords` write for that case. For `nreqs>1`, `cmplsRecords` is still required to return per-buffer sizes.

**What to put in `size_q` when `nreqs > 1`?**

Always set `size_q = MPIB_IMM_SIZEQ_SENTINEL`.

Rationale: for `nreqs>1` the receiver must read *multiple* sizes from `cmplsRecords[slot][i]`, so `size_q` is not used for payload size and should explicitly indicate “consult `cmplsRecords`”.

**Why keep an explicit `active_mask` at all?**

Even with `cmplsRecords`, the receiver needs to know *how many IMMs to wait for* to declare completion. `nreqs` does not encode “how many rails were used”, and a total size does not imply a single rail (a split transfer can still have `nreqs==1`).

An explicit `active_mask` is the simplest way to:
- Avoid dummy 0-byte IMMs on inactive rails.
- Prevent receiver-side queue drift while still requiring one IMM per active rail.
- Let the receiver learn its `expected_mask` on the first arriving IMM.

#### 2.3 Receiver State Machine (Mask Learning)
The receiver cannot know "how many completions to expect" in advance. It learns dynamically:

1.  **Event:** CQE arrives with `imm_data`.
2.  **Decode:** Extract `slot` and `mask`. Map `slot` to local `mpibRequest`.
3.  **Learn:** If `req->expected_mask` is 0 (Unset), set `req->expected_mask = imm.active_mask`.
4.  **Progress:** Mark this rail as seen (`req->seen_mask |= (1 << dev)`).
5.  **Completion:** When `req->seen_mask == req->expected_mask`, the request is complete.

---

## Implementation Checklist

### Phase 1: Transport & Init
- [x] Initialize `ibv_srq` per `mpibNetCommDevBase`.
- [x] Update `ibv_create_qp` to use SRQ (`.srq = dev->srq`, `.max_recv_wr = 0`).
- [x] Define `HighWater` / `LowWater` marks for SRQ depth. (Hardcoded: 64/512)
- [x] Add `ibv_post_srq_recv()` wrapper usage in SRQ refill path.

### Phase 2: Sender Logic (`mpibIsend`)
- [x] Compute `active_mask` dynamically (0x1, 0x2, or 0x3).
- [x] Post payload RDMA WRs only on devices present in `active_mask`.
- [x] Post exactly one signaling `WR_RDMA_WRITE_WITH_IMM` per active device.
- [x] On the *leader rail only*, attach an SGE to the signaling WR to write sizes into `remCmplsRecords[slot]`.
- [x] Encode `(slot | mask << 8 | size_q << 10)` in `imm_data`.
- [x] Handle 0-byte transfers: force `active_mask = 0x1` when all sizes are 0.

### Phase 3: Receiver Logic (`mpibIrecv` / `mpibTest`)
- [x] **Remove** loop calling `ibv_post_recv` per request.
- [x] Add `mpibSrqCheckAndRefill(comm)` to `mpibIrecv` and `mpibTest`.
- [x] Add assertions for protocol violations (slot reuse, NULL slotReq, invalid active_mask).

### Phase 4: Completion Handler
- [x] Rewrite `mpibCompletionEventProcess`:
    - [x] Remove `wr_id` dependency for RECV completions.
    - [x] Decode `imm_data` (slot, active_mask, size_q).
    - [x] Implement Mask Learning logic (expected_mask/seen_mask).
    - [x] Decrement `srqPosted` on each RECV CQE.

### Phase 5: Future Optimizations (NOT YET IMPLEMENTED)
- [ ] **size_q fast-path for `nreqs==1`**: When size <= 512MiB, encode actual size in `size_q` instead of SENTINEL, skip cmplsRecords write, and merge data+signaling into 1 WR (instead of current 2 WRs).

---

## Current Implementation Notes

**WR structure (as implemented):**
- For all `nreqs` values: `wrs[0..nreqs-1]` = data WRs + `wrs[nreqs]` = signaling WR
- This matches the design's "overflow path" since we always use `size_q = SENTINEL`
- The extra signaling WR is required because data goes to `slots[r].addr` while cmplsRecords goes to `remCmplsRecords.addr` (different remote addresses)
- Non-leader active rails post the signaling WR with `num_sge=0` (IMM-only)

**Verified working:**
- SRQ-based receive with mask learning
- Dynamic rail skipping (only active rails post/receive)
- 0-byte transfer handling (force SOUT active)

---

## Detailed Code Change Plan

This section maps the design to concrete edits in MPIB (by file and symbol). It is written to keep the implementation **clean** (localized changes, minimal new state) and **efficient** (batched SRQ posting, skip inactive rails entirely).

### 1) Add SRQ plumbing (verbs wrappers)

**Files:**
- `ext-net/mpib/src/ibvsymbols.h`
- `ext-net/mpib/src/ibvsymbols.cc`
- `ext-net/mpib/src/ibvwrap.h`
- `ext-net/mpib/src/ibvwrap.cc`

**Edits (minimize new names; keep call sites in existing functions):**
- Add symbol pointers to `struct ncclIbvSymbols` for:
    - `ibv_create_srq`
    - `ibv_destroy_srq`
- In `buildIbvSymbols(...)` (RDMA-core path in `ibvsymbols.cc`), `ASSIGN_SYM(...)` those symbols.
 - Add wrappers:
     - `wrap_ibv_create_srq(struct ibv_srq** ret, struct ibv_pd* pd, struct ibv_srq_init_attr* attr)`
     - `wrap_ibv_destroy_srq(struct ibv_srq* srq)`
     - `wrap_ibv_post_srq_recv(struct ibv_srq* srq, struct ibv_recv_wr* wr, struct ibv_recv_wr** bad_wr)`

Notes:
- `wrap_ibv_post_srq_recv` should mirror the existing style of `wrap_ibv_post_recv`/`wrap_ibv_post_send`. It can call `srq->context->ops.post_srq_recv(...)` (no extra dlsym needed).

### 2) Extend comm/request structs for SRQ + slot mapping

**Files:**
- `ext-net/mpib/src/mpib_common.h`
- `ext-net/mpib/src/mpib_p2p.cc` (request init/reset)
- `ext-net/mpib/src/mpib_connect.cc` (base init/destroy)

**Edits (minimal, no ABI exposure outside plugin):**
- In `struct mpibNetCommDevBase` (in `mpib_common.h`), add SRQ fields:
    - `struct ibv_srq* srq;` (NULL for send comms; non-NULL for recv comms)
    - `int srqPosted;` (how many generic SRQ WQEs we believe are posted)
    - `int srqLowWater, srqHighWater;` (watermarks for refill)
- In `struct mpibNetCommBase`, add a slot→request map used by the SRQ-based completion path:
    - `struct mpibRequest* slotReq[NET_IB_MAX_REQUESTS];`
- In `struct mpibRequest`, add per-recv request completion tracking:
    - `uint8_t slot;` (0..255)
    - `uint8_t expected_mask;` (0 until first IMM arrives)
    - `uint8_t seen_mask;` (OR of rails that have delivered an IMM)

**Initialization points:**
- `mpibGetRequest(...)` in `ext-net/mpib/src/mpib_p2p.cc` should clear `slot/expected_mask/seen_mask` for reuse.
- In `mpibInitCommDevBase(...)` in `ext-net/mpib/src/mpib_connect.cc`, set `base->srq = NULL` and initialize SRQ counters to 0.
- In `mpibDestroyBase(...)`, if `base->srq != NULL`, destroy it before destroying the CQ/PD.

### 3) Create one SRQ per device for recv comms (and attach SRQ to QPs)

**Files:**
- `ext-net/mpib/src/mpib_connect.cc`

**Edits:**
- Add a helper (static) such as `mpibCreateSrqForRecvBase(struct mpibNetCommDevBase* base)` that:
    - Creates `base->srq` with `wrap_ibv_create_srq`.
    - Picks SRQ depth/watermarks. A practical starting point:
        - `srqHighWater = 2 * NET_IB_MAX_REQUESTS`
        - `srqLowWater  = NET_IB_MAX_REQUESTS`
    - Uses `ibv_srq_init_attr` with `attr.max_wr = srqHighWater` and `attr.max_sge = 1` (even though we post `num_sge=0`).
- In `mpibAccept(...)`, after each `mpibInitCommDevBase(...)` for `rComm->devs[i].base`, call `mpibCreateSrqForRecvBase(...)`.
- In `mpibCreateQp(...)`:
    - If `base->srq != NULL`, set `qpInitAttr.srq = base->srq`.
    - (Optional but recommended) reduce per-QP receive caps since SRQ is used:
        - keep `qpInitAttr.cap.max_recv_wr` small or 0 (driver-dependent); `max_recv_sge` can remain 1.

This ensures that **all receiver QPs** (created inside `mpibReceiverQpsCreateToRts(...)`) pull receives from the SRQ.

### 4) Add SRQ refill helper (generic receives)

**Files:**
- `ext-net/mpib/src/mpib_p2p.cc`

**New helper:**
- `static ncclResult_t mpibSrqCheckAndRefill(struct mpibNetCommBase* base)`
    - No-op if `base->isSend`.
    - For each `devIndex` in `[0..base->vProps.ndevs)`:
        - `mpibNetCommDevBase* devBase = mpibGetNetCommDevBase(base, devIndex)`
        - If `devBase->srq == NULL`, continue.
        - If `devBase->srqPosted >= devBase->srqLowWater`, continue.
        - Post a batch of SRQ receives (generic, 0 SGE):
            - `ibv_recv_wr wr; wr.wr_id=0; wr.sg_list=NULL; wr.num_sge=0; wr.next=...`
            - Call `wrap_ibv_post_srq_recv(devBase->srq, &wr, &bad_wr)` repeatedly until `srqPosted == srqHighWater`.
            - Update `srqPosted` only on successful posts.

### 5) Change `mpibIrecv` to stop posting per-QP receives, and record slot mapping

**Files:**
- `ext-net/mpib/src/mpib_p2p.cc`

**Edits in `mpibIrecv(...)`:**
- Before posting CTS (`mpibPostFifo(...)`), compute the slot that will be used:
    - `uint8_t slot = comm->base.fifoHead % NET_IB_MAX_REQUESTS;`
- Set request state:
    - `req->slot = slot; req->expected_mask = 0; req->seen_mask = 0;`
    - `comm->base.slotReq[slot] = req;` (debug-check it was NULL)
- Call `mpibSrqCheckAndRefill(&comm->base)`.
- Delete the loop that currently does `wrap_ibv_post_recv(qp->qp, ...)` per device.
- Do **not** call `mpibAddEvent(req, ...)` for data-IMM progress anymore; completion is driven by `expected_mask/seen_mask`.

### 6) Change `mpibIsend` to compute `active_mask` and skip inactive rails entirely

**Files:**
- `ext-net/mpib/src/mpib_p2p.cc`

**Edits in `mpibIsend(...)`:**
- After computing `sizeSout[i]`/`sizeSup[i]` per request, compute a **per-slot** `active_mask`:
    - bit0 set if any `sizeSout[j] > 0`
    - bit1 set if any `sizeSup[j] > 0`
- Select leader rail from `active_mask`.
- Update the initial “Populate events per QP” loop to only add events for devices present in `active_mask`.
- Update the “Post WRs” loop to only post to rails present in `active_mask`.
- Set `lastWr->imm_data` to the packed metadata `(slot, active_mask, size_q)` (network byte order).
- For `nreqs > 1` (size side-channel):
    - Only the leader rail attaches the size-record SGE to the signaling WR.
    - Non-leader active rails send IMM-only (`lastWr->num_sge = 0; lastWr->sg_list = NULL`).

**Staging recommendation (keeps first implementation simple; strongly recommended):**
- **MUST (first implementation):** always set `size_q = MPIB_IMM_SIZEQ_SENTINEL`.
- **MUST (first implementation):** always write `cmplsRecords` (leader-only) for all `nreqs >= 1`.
- Only after correctness is validated, add the `size_q` fast-path optimization for `nreqs==1`.

### 7) Rewrite completion dispatch for recv IMMs (slot/mask driven)

**Files:**
- `ext-net/mpib/src/mpib_p2p.cc`

**Edits in `mpibCompletionEventProcess(...)`:**
- Keep the existing SEND completion behavior (using `wc->wr_id` packing).
- For `IBV_WC_RECV_RDMA_WITH_IMM`:
    - Decode `slot`, `active_mask`, `size_q` from `be32toh(wc->imm_data)`.
    - Lookup `mpibRequest* req = commBase->slotReq[slot]` and validate it is a RECV request.
    - If `req->expected_mask == 0`, set it to `active_mask`.
    - Set `req->seen_mask |= (1u << devIndex)`.
    - If `req->nreqs == 1` and `size_q != MPIB_IMM_SIZEQ_SENTINEL`, set `req->recv.sizes[0]` from `size_q` (with clamping).
    - Decrement `devBase->srqPosted` (each CQE consumes one SRQ WQE).

**Edits in `mpibTest(...)`:**
- Ensure progress even when a particular RECV request has no per-device `events[]`:
    - For RECV requests, poll both device CQs for the parent comm (same as today, but not gated on `r->events[i]`).
    - After polling, check completion using mask state: `done` when `expected_mask != 0 && seen_mask == expected_mask`.
    - Call `mpibSrqCheckAndRefill(r->base)` periodically (safety refill point).

**Edits in request completion path:**
- When a RECV request completes:
    - Clear `base->slotReq[req->slot] = NULL`.
    - Return sizes from `req->recv.sizes[]` as today.

### 8) Add lightweight bring-up checks

**Files:**
- `ext-net/mpib/src/mpib_p2p.cc`

**Debug checks (behind existing logging/assert style):**
- In `mpibIrecv`: assert/validate `base.slotReq[slot] == NULL` before setting (detect slot reuse while outstanding).
- In completion decode: validate `active_mask != 0` and `active_mask` is a subset of `(1<<base->vProps.ndevs)-1`.
- If an IMM arrives for a slot with `slotReq[slot] == NULL`, count as protocol error (return `ncclInternalError`).

### 9) (Optional) documentation-to-code consistency note

After the above lands, update any comments in `mpib_p2p.cc` that still mention “one RECV per QP per request” so they reflect “SRQ per device + slot/mask via IMM”.





---

## Discussion: Rejected Alternatives

### Option 1: Naive Rail Skipping (No SRQ)
*Proposal:* Keep standard RQs. If sender decides 100/0 split, it just skips sending on the inactive rail.
*   **Why it fails:** This causes **"Ghost WQEs"**. The receiver posts a WQE on the inactive rail (expecting a potential message). If the message is skipped, that WQE remains at the head of the queue. It will be incorrectly consumed by the *next* message targeting that rail, causing catastrophic `wr_id` / data misalignment.

### Option 2: CTS-Based Sharding
*Proposal:* Validating the active rails during the CTS (Clear-To-Send) handshake. Receiver decides the split and tells Sender.
*   **Why it fails:** **Stale Telemetry**.
    *   CTS (receiver decision) happens at $T_0$ (receiver `irecv` time).
    *   Data transmission (sender action) happens later at $T_1$ (sender `isend` time), and the gap is driven by NCCL’s pipeline depth / scheduling. It is **not** bounded by RTT and can be orders of magnitude larger than RTT.
    *   If congestion telemetry is fast-changing, deciding routing at $T_0$ to guide traffic at $T_1$ can be stale. Sender-side hints sampled at $T_1$ are fresher.

### Option 3: Standard Size in IMM
*Proposal:* Keeping the message size in `imm_data` (current behavior).
*   **Why it fails:** We need `imm_data` to carry the **Routing Protocol** (Slot ID + Active Mask) because SRQ removes the ability to associate WQEs with Request IDs implicitly. 32 bits is not enough for both Protocol and Size.

