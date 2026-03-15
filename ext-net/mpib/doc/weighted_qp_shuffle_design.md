# Weighted QP Shuffle Design

## Summary

MPIB data transfer uses **one rail / one QP per message**, chosen by the receiver using a **SUP-bandwidth-weighted shuffle**. The selection is communicated to the sender via CTS metadata. Completions are matched by `wr_id`, mirroring net-ib semantics.

This replaces the previous split-message / SRQ design, which required slot/mask protocol state to reconstruct receive completions.

---

## Design Goals

- One message → one QP → one receive completion
- Regular RQ in the data path; `wr_id`-based request identity
- Two-rail utilization preserved statistically over many messages via `supbw`-weighted shuffle
- No `expected_mask`/`seen_mask`; no `slotReq[]`

---

## Rail/QP Selection

**Owner:** Receiver (`mpibIrecv`).

The receiver selects a rail and QP before posting the RECV, then advertises the choice in CTS. The sender reads CTS and posts on exactly that QP. This is required because with regular RQ the receiver must know the target QP before the sender transmits.

### `mpibGetSupBw(mpibRecvComm *comm, size_t size)`

- `MPIB_MODE=0` (vanilla): returns `UINT32_MAX` (intra-island, SUP-only) or `0` (inter-island, SOUT-only). No SHM read.
- `MPIB_MODE=1` (advanced): reads agent hint from SHM via seqlock. `size` reserved for future BDP threshold.

### `mpibWeightedSelectQp()`

Returns flat `qps[]` index and `*outDevIndex`.

| supbwHint | Behavior |
|-----------|----------|
| `0` or `nqpsSup == 0` | SOUT-only, round-robin within SOUT QPs |
| `≥ 1024` | SUP-only, round-robin within SUP QPs |
| `1..1023` | `totalCursor++ % 1024 < supbwHint` → SUP, else SOUT |

`supbwHint` is parts-per-1024 of SUP's share (e.g., 512 = equal split). Since the hint is re-read on every `mpibIrecv`, the split ratio adapts immediately when the agent updates.

**Cursor state** (recv-side only, on `mpibNetCommBase`):
- `qpCursorSout` / `qpCursorSup` — round-robin within each rail's QPs
- `totalCursor` — monotonic message counter for the weighted split

---

## CTS Metadata

`mpibSendFifo` carries the receiver's selection:

```c
uint8_t selectedDevIndex;   // 0 = SOUT, 1 = SUP
uint8_t selectedQpIndex;    // flat index into base.qps[]
```

Fits in existing padding; struct stays `alignas(32)`.

---

## Send Path

1. Read `selectedDevIndex` / `selectedQpIndex` from CTS.
2. Build WR chain on that single QP — no split, no `active_mask`.
3. `lastWr` is `IBV_WR_RDMA_WRITE_WITH_IMM`; `imm_data = htobe32(reqs[0]->send.size)`.
4. `nreqs > 1`: attach cmplsRecords SGE to `lastWr`. `nreqs == 1`: no cmplsRecords RDMA (size carried in IMM, mirrors net-ib).
5. Post once: `ibv_post_send(selectedQp->qp, wrs, &bad_wr)`.

---

## Receive Path

1. `mpibGetSupBw()` → `mpibWeightedSelectQp()` → `selectedQpIdx`, `selectedDevIndex`.
2. Post one `ibv_post_recv(selectedQp->qp)` — regular RQ, `wr_id = req - base->reqs`.
3. `mpibAddEvent(req, selectedDevIndex)` — data RECV event.
4. `mpibPostFifo()` — writes CTS with selection; adds CTS signaling event (may be on a different device).

---

## Completion Model

### Reference

`src/transport/net_ib/p2p.cc` — net-ib patterns adopted verbatim:
- `wr_id = req - comm->base.reqs` (index, not pointer)
- `req->events[devIndex]--` on CQE; complete when all slots zero
- `devBases[devIndex]` array iteration in test loop

### Events tracking

```c
int events[MPIB_MAX_DEVS];
struct mpibNetCommDevBase *devBases[MPIB_MAX_DEVS];
```

Arrays are necessary — CTS signaling (`mpibPostFifo`) can fire on a different device than the data RECV, so one request can have concurrent events on two devices.

`mpibAddEvent(req, devIndex)` — increments `events[devIndex]++`, sets `devBases[devIndex]`.

### RECV completion (`IBV_WC_RECV_RDMA_WITH_IMM`)

- Lookup: `req = reqs[wc->wr_id & 0xff]`
- `nreqs == 1`: `req->recv.sizes[0] = be32toh(wc->imm_data)`
- `nreqs > 1`: sizes in cmplsRecords, written by sender's `lastWr`
- Decrement: `req->events[devIndex]--`

### `mpibTest()` polling

```c
for (int i = 0; i < MPIB_MAX_DEVS; i++) {
    if (r->devBases[i] == NULL || r->events[i] == 0) continue;
    poll r->devBases[i]->cq → mpibCompletionEventProcess(base, wc, i);
}
```

---

## What Was Removed

| Removed | Replaced by |
|---------|-------------|
| Split-message across rails | Single QP per message |
| `active_mask` / `expected_mask` / `seen_mask` | `wr_id`-based lookup |
| `slotReq[slot]` | `req - base->reqs` index in `wr_id` |
| SRQ + `ibv_post_srq_recv` | Regular RQ on selected QP |
| IMM slot/mask encoding | `imm_data = reqs[0]->send.size` |
| Leader/non-leader cmplsRecords logic | Single unconditional write (when `nreqs > 1`) |
