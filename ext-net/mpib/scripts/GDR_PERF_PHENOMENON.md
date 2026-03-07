# NCCL GDR Performance Phenomenon (Current Investigation)

Date: 2026-03-04  
Scope: 8 nodes (`c1`..`c8`), 1 GPU per node, `all_reduce_perf` over RoCE/IB using NCCL netib path.

## 1) What phenomenon we are seeing

In this cluster, enabling GPU Direct RDMA (GDR) does **not** improve all-reduce throughput.

Observed outcomes from recent runs:

- `baseline.log` (non-GDR path):
  - Avg bus bandwidth: **8.96267 GB/s**
  - 1 GiB busbw: **18.03 / 17.61 GB/s** (out-of-place / in-place)
- `gdr-read0.log` (GDR enabled, read=0):
  - Avg bus bandwidth: **4.54637 GB/s**
  - 1 GiB busbw: **7.92 / 7.93 GB/s**
- `gdr-read1.log` (GDR enabled, read=1 requested; mixed read behavior appears in logs):
  - Avg bus bandwidth: **7.62309 GB/s**
  - 1 GiB busbw: **13.23 / 13.60 GB/s**

So, in this environment:

- `baseline` > `gdr-read1` > `gdr-read0`

This is opposite of the expected “GDR always faster” intuition.

---

## 2) What we have tried so far

### A. Per-node NIC/HCA mapping cleanup

- Reworked launch path to call a rank wrapper (`rank_wrapper.sh`) per rank.
- Standardized per-node static mapping via `host_map.conf`:
  - node hostname
  - `pf0/pf1` NIC names
  - `mlx5_*` mapping for `SOUT/SUP`
  - one chosen CPU core per node
- Removed hardcoded global NIC assumptions from launcher.

### B. Host selection cleanup

- Corrected host usage to rely on `hostfile` (`c1..c8`) and removed stale `vm*` behavior.

### C. CPU affinity / pinning changes

- Added Open MPI controls in launcher:
  - `--bind-to ${MPI_BIND_TO}`
  - `--map-by ${MPI_MAP_BY}`
  - optional `--report-bindings`
- Added explicit rank pinning in `rank_wrapper.sh` via `taskset -pc <cpu_core>`.
- Current static core map (from `host_map.conf`):
  - `c1->1, c2->16, c3->32, c4->48, c5->1, c6->32, c7->64, c8->96`
- Logs confirm proxy service threads follow these cores.

### D. Transport/path controls to isolate cross-node behavior

- Forced local transports off for debugging consistency:
  - `NCCL_SHM_DISABLE=1`
  - `NCCL_P2P_DISABLE=1`
  - `NCCL_GIN_TYPE=0`
- Confirmed network path is netib (`Using network IB`) with selected per-node `NCCL_IB_HCA`.

### E. GDR knob matrix attempts

- Tested baseline and GDR settings via:
  - `NCCL_NET_GDR_LEVEL`
  - `NCCL_NET_GDR_READ`
- Verified runs by checking runtime log lines such as:
  - `GPU Direct RDMA Enabled for GPU...` vs
  - `GPU Direct RDMA Disabled for GPU...`

---

## 3) Key observations from logs

1. NCCL detects GDR capability at HCA level (`GPU Direct RDMA Enabled for HCA...`) but that does not guarantee the final GPU↔NET path is used for data transfers in every mode.
2. Some runs clearly show GPU-level GDR disabled by distance policy (e.g., `distance ... > ...`) depending on `NCCL_NET_GDR_LEVEL` setting.
3. Even in runs where GPU-level GDR is enabled, end-to-end all-reduce busbw can be lower than baseline in this cluster.
4. CPU pinning and host/NIC mapping are now stable and deterministic, so the remaining gap is less likely due to accidental rank placement or wrong NIC names.

---

## 4) Current script state (important)

The launcher currently has:

- `NCCL_NET_GDR_LEVEL=6`
- `NCCL_NET_GDR_READ=${NCCL_NET_GDR_READ:-1}`

The nearby comment saying baseline mode may be stale relative to those values.

---

## 5) Working hypothesis

This looks like a topology/policy-sensitive GDR behavior where:

- GDR is not universally beneficial for this specific GPU↔NIC path and collective pattern.
- `read=0` is especially poor here.
- `read=1` partially recovers performance but still trails the measured baseline.

---

## 6) Practical takeaway for now

For this exact test setup (8 nodes, 1 GPU/node, current topology and knobs), **baseline non-GDR currently gives the best observed throughput** among tested modes.

If we continue tuning, prioritize reproducible A/B runs with fixed launcher state and clear per-run logging of:

- `NCCL_NET_GDR_LEVEL`
- `NCCL_NET_GDR_READ`
- final `GPU Direct RDMA Enabled/Disabled for GPU...` lines
- 1 GiB busbw and Avg bus bandwidth
