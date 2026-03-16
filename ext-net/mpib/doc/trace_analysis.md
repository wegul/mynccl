# MPIB Network Trace Analysis

## 1. What We Instrument

Two TRACE points in `src/mpib_p2p.cc`, both **sender-side only** (no cross-node clock sync needed):

| Event | Function | When it fires |
|---|---|---|
| `post_send` | `mpibIsend()` | WQE submitted to HW send queue |
| `cqe_send` | `mpibCompletionEventProcess()` | Send CQE harvested from CQ |

Because the opcode is `IBV_WR_RDMA_WRITE_WITH_IMM`, the send CQE is only generated **after the remote side ACKs** — so `post_send → cqe_send` measures true round-trip latency, not just local posting.

## 2. Trace Line Format

```
<host>:<pid>:<tid> [<gpu>] <timestamp_ms> <file>:<line> NCCL TRACE <payload>
```

### `post_send` payload
```
event=post_send req=<slot> size=<bytes> nreqs=<N> dev=<0|1> qp=<idx> path=<0|1>
```

### `cqe_send` payload
```
event=cqe_send req=<slot> size=<bytes> dev=<0|1>
```

### Field reference

| Field | Type | Meaning |
|---|---|---|
| `host` | string | Hostname (e.g., `c5`) |
| `pid` | int | Process ID |
| `tid` | int | Thread ID |
| `timestamp_ms` | float | Milliseconds since NCCL epoch (process-local, monotonic) |
| `req` | int | Request slot index (0..NET_IB_MAX_REQUESTS-1), correlation key |
| `comm` | uint | `commBase` pointer as unsigned long, unique per communicator per process |
| `size` | int | Message size in bytes |
| `nreqs` | int | Sub-requests in this batch (usually 1) |
| `dev` | int | 0 = SOUT NIC, 1 = SUP NIC |
| `qp` | int | QP index (flat index into `base.qps[]`) |
| `path` | int | 0 = intra-island, 1 = cross-island |

## 3. Timeline Map

```
RECEIVER                                        SENDER
───────────────────────────────────────────────────────────────────────
mpibIrecv()
  ibv_post_recv(QP)
  ┌──────────────────────────────┐
  │ post_recv (not traced)       │
  └──────────────────────────────┘
  mpibPostFifo()  ──RDMA_WRITE──────────────→  ctsFifo[slot].idx = N
  ┌──────────────────────────────┐
  │ post_cts  (not traced)       │              (mpibIsend() busy-polls
  └──────────────────────────────┘               slots[0].idx until match)
  return req                                            ↓ CTS lands
                                                 ┌─────────────────────────────┐
                                                 │ event=post_send req=3 ...   │ ← TRACED
                                                 └─────────────────────────────┘
                                                 ibv_post_send()
                                                   ──RDMA_WRITE_WITH_IMM──→

═══════════════════════════ wire RTT ═══════════════════════════════════

  mpibTest() → ibv_poll_cq()                     mpibTest() → ibv_poll_cq()
  ┌──────────────────────────────┐              ┌─────────────────────────────┐
  │ cqe_recv  (not traced)       │              │ event=cqe_send req=3 ...    │ ← TRACED
  └──────────────────────────────┘              └─────────────────────────────┘
  req.events[D]-- → done                         req.events[D]-- → done
```

## 4. Correlation Key

Match `post_send` ↔ `cqe_send` by the tuple `(host, pid, comm, req)`.

- `comm` is the `commBase` pointer cast to `unsigned long` — unique per communicator object within a process. This is critical in collectives like alltoall where a single rank has **N-1 send communicators** all running in the same process (same `host:pid`). Without `comm`, slots from different communicators collide on the same `req` index.
- `req` is the slot index (0..NET_IB_MAX_REQUESTS-1) within that communicator.

The `req` slot is recycled after completion, so process the log **sequentially** and pair each `post_send` with the **next** `cqe_send` for the same key.

## 5. Four-Way Case Classification

Derived from `post_send` fields:

| `path=` | `dev=` | Case Label |
|---|---|---|
| 0 | 0 | `intra-sout` |
| 0 | 1 | `intra-sup` |
| 1 | 0 | `cross-sout` |
| 1 | 1 | `cross-sup` |

## 6. Analysis Tables

### Stage 1 — Flat Table (one row per message, from parser)

This is the raw joined output, saved as CSV. Every subsequent table and plot
derives from it.

| host | pid | req | case | size | dev | path | qp | latency_ms | xput_gbps |
|------|-----|-----|------|------|-----|------|----|------------|-----------|
| c5 | 9348 | 3 | cross-sout | 131072 | 0 | 1 | 2 | 0.312 | 3.37 |
| c5 | 9348 | 4 | intra-sup  | 131072 | 1 | 0 | 5 | 0.089 | 11.79 |
| c8 | 8278 | 1 | cross-sout | 65536  | 0 | 1 | 2 | 0.298 | 1.76 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

### Stage 2 — Latency Summary Table (grouped by case × size bucket)

Aggregate p50/p99/max latency per case per message size. Size is bucketed
into the standard nccl-tests powers-of-two (8K, 16K, ..., 1G).

| case | size | count | p50_ms | p95_ms | p99_ms | max_ms |
|------|------|-------|--------|--------|--------|--------|
| intra-sout | 131072 | 1842 | 0.091 | 0.103 | 0.118 | 0.247 |
| intra-sup  | 131072 | 1203 | 0.085 | 0.097 | 0.109 | 0.201 |
| cross-sout | 131072 | 2011 | 0.312 | 0.341 | 0.389 | 0.821 |
| cross-sup  | 131072 |  893 | 0.298 | 0.330 | 0.371 | 0.704 |
| ... | ... | ... | ... | ... | ... | ... |

→ Feeds plots: **latency vs. message size** lines (one per case), and
  **latency CDF** per case at a fixed size.

### Stage 3 — Throughput Summary Table (grouped by case × time window)

Slice the flat table into fixed-width time windows (e.g., 100ms), sum bytes
completed in each window, convert to Gbps.

| window_start_ms | case | bytes_completed | agg_xput_gbps |
|-----------------|------|-----------------|---------------|
| 0.0 | intra-sout | 2684354560 | 214.7 |
| 0.0 | cross-sout | 1073741824 | 85.9 |
| 100.0 | intra-sout | 2415919104 | 193.3 |
| ... | ... | ... | ... |

→ Feeds plot: **throughput over time** per case (line chart), and
  **avg throughput bar chart** per case.

### Stage 4 — Per-Message Throughput Table (for distribution plots)

One row per message, throughput already computed in Stage 1 (`xput_gbps`).
Group by `(case, size)` and compute percentiles the same way as latency.

| case | size | count | p50_gbps | p95_gbps | p99_gbps |
|------|------|-------|----------|----------|----------|
| intra-sout | 131072 | 1842 | 11.5 | 10.2 | 9.6 |
| cross-sout | 131072 | 2011 |  3.4 |  2.9 | 2.4 |
| ... | ... | ... | ... | ... | ... |

→ Feeds plot: **throughput CDF** and **box plots** per case.

---

## 7. Analysis Plan

### 7.1 Message Size Distribution

- Source: Stage 1 flat table, `post_send` rows only
- Group by: `(case)`
- Plot: histogram of `size` per case

### 7.2 Completion Latency (avg / tail)

- Source: Stage 2 latency summary table
- Compute: `latency_ms = cqe_send.ts - post_send.ts`
- Group by: `(case, size)`
- Plot: p50 / p95 / p99 / max per group
- Note: both timestamps are from the same process — no clock sync needed

### 7.3 Aggregate Throughput

- Source: Stage 3 throughput summary table
- Slice into fixed time windows (e.g., 100ms), sum bytes per `(case, window)`:
  ```
  agg_xput = sum(size for all cqe_send in [t0,t1]) / (t1 - t0)
  ```
