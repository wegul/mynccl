# NCCL RoCE performance analysis (GDR vs non-GDR, binding, and root-cause sweep)

## Scope and artifacts

### Baseline A/B sweep (5 trials each)
- logs: [ext-net/mpib/scripts/log](ext-net/mpib/scripts/log)
- summary: [ext-net/mpib/scripts/log/summary.md](ext-net/mpib/scripts/log/summary.md)
- csv: [ext-net/mpib/scripts/log/summary.csv](ext-net/mpib/scripts/log/summary.csv)

### Root-cause sweep (3 trials per config)
- logs: [ext-net/mpib/scripts/log/rootcause](ext-net/mpib/scripts/log/rootcause)
- summary: [ext-net/mpib/scripts/log/rootcause/rootcause_summary.md](ext-net/mpib/scripts/log/rootcause/rootcause_summary.md)
- csv: [ext-net/mpib/scripts/log/rootcause/rootcause_summary.csv](ext-net/mpib/scripts/log/rootcause/rootcause_summary.csv)

Launcher and runners used:
- [ext-net/mpib/scripts/launch_nccl.sh](ext-net/mpib/scripts/launch_nccl.sh)
- [ext-net/mpib/scripts/run_baseline_gdr_vs_nogdr.sh](ext-net/mpib/scripts/run_baseline_gdr_vs_nogdr.sh)
- [ext-net/mpib/scripts/run_rootcause_gdr_sweep.sh](ext-net/mpib/scripts/run_rootcause_gdr_sweep.sh)

Workload (all runs): `all_reduce_perf -b 8 -e 1G -f 2 -g 1 -c 1` on RoCE `mlx5_0`.

## 1) Baseline result (clean GDR vs non-GDR)

From [ext-net/mpib/scripts/log/summary.md](ext-net/mpib/scripts/log/summary.md):

- Avg bus BW mean
  - GDR: **5.93530**
  - non-GDR: **5.91570**
  - delta: **+0.33%** (tiny)

- 1 GiB out-of-place mean
  - GDR: **15.770 GB/s**
  - non-GDR: **16.484 GB/s**
  - delta: **-4.33%** (GDR lower)

- 1 GiB in-place mean
  - GDR: **15.720 GB/s**
  - non-GDR: **16.430 GB/s**
  - delta: **-4.32%** (GDR lower)

So the earlier conclusion still holds: GDR is not winning at 1 GiB in this setup.

## 2) Binding policy analysis (bind-none vs bind-core vs bind-numa)

This addresses your point directly.

From [ext-net/mpib/scripts/log/rootcause/rootcause_summary.md](ext-net/mpib/scripts/log/rootcause/rootcause_summary.md):

### GDR mode
- `bind_none_gdr`: 1 GiB in-place **15.777 GB/s**
- `bind_core_gdr`: 1 GiB in-place **15.800 GB/s**
- `bind_numa_gdr`: 1 GiB in-place **15.760 GB/s**

Spread is very small (about $0.04$ GB/s, i.e. sub-1%).

### non-GDR mode
- `bind_none_nogdr`: 1 GiB in-place **16.127 GB/s**
- `bind_core_nogdr`: 1 GiB in-place **16.397 GB/s**
- `bind_numa_nogdr`: 1 GiB in-place **16.203 GB/s**

non-GDR also varies modestly with binding. Core-bind is best among tested policies, but not enough to explain a major GDR gap.

### Verdict on binding
Binding does matter slightly, but it does **not** flip the main conclusion. GDR remains below non-GDR for 1 GiB under all tested binding policies.

## 3) Other tested issues that can hinder GDR

### A) `NCCL_NET_GDR_READ`
- `gdrread1_gdr`: 1 GiB in-place **15.820 GB/s**, GDR markers `[1,1,1]`
- `gdrread0_gdr`: 1 GiB in-place **9.343 GB/s**, GDR markers `[0,0,0]`

This is a critical finding: with `NCCL_NET_GDR_READ=0`, the logs show GDR disabled (`GDR 0`) and performance collapses. This can silently invalidate “GDR” runs if not checked.

### B) `NCCL_IB_QPS_PER_CONNECTION`
- `qps1_gdr`: 1 GiB in-place **15.807 GB/s**
- `qps2_gdr`: 1 GiB in-place **13.273 GB/s**
- `qps4_gdr`: 1 GiB in-place **13.087 GB/s**

In this environment, increasing QPs hurts throughput significantly. Default-like `qps=1` is best among tested values.

### C) Channel count stability
All tested runs reported `2 coll channels` in logs (see root-cause summary), so no evidence here that channel-count variance is causing the GDR underperformance.

## 4) Why still far from 200G-equivalent

For 200 Gb/s link speed, idealized line-rate is:

$$
\frac{200\ \text{Gb/s}}{8} = 25\ \text{GB/s}
$$

Best observed 1 GiB numbers in these sweeps are around $16.3$–$16.4$ GB/s (non-GDR), i.e. roughly $65\%$ of 25 GB/s.

So your concern is valid: current stack is not near wire-rate-equivalent throughput.

## Final conclusions
1. Binding (`core` / `numa` / `none`) has only secondary impact.
2. GDR is still not better than non-GDR at 1 GiB in current setup.
3. `NCCL_NET_GDR_READ=0` is a high-risk confounder; it effectively kills GDR path in these tests.
4. `NCCL_IB_QPS_PER_CONNECTION>1` degraded performance here.
5. Remaining gap to 200G-equivalent is substantial and requires deeper transport/system profiling.

## Recommended next step (targeted)
1. Keep `NCCL_NET_GDR_READ=1` and verify `GDR 1` every run.
2. Keep `NCCL_IB_QPS_PER_CONNECTION=1` for this platform.
3. Fix binding policy (pick one, e.g., `--bind-to core`) and keep it identical across A/B.
4. Add NIC/PCIe counters during runs (mlx5 and host PCIe) to localize whether bottleneck is NIC queueing, PCIe, or software path.
