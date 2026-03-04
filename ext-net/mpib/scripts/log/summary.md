# GDR vs non-GDR run summary

- Trials per mode: 5
- Metrics extracted from nccl-tests `all_reduce_perf` logs

## Aggregated metrics

| mode | avg bus BW mean | avg bus BW p50 | avg bus BW p90 | 1GiB out-of-place algBW mean | 1GiB in-place algBW mean | GDR marker values |
|---|---:|---:|---:|---:|---:|---|
| gdr | 5.93530 | 5.93379 | 5.95159 | 15.770 | 15.720 | [1, 1, 1, 1, 1] |
| nogdr | 5.91570 | 5.89517 | 5.95219 | 16.484 | 16.430 | [0, 0, 0, 0, 0] |

## Trial details

| mode | trial | avg bus BW | 1GiB out algBW | 1GiB in algBW | GDR marker |
|---|---:|---:|---:|---:|---:|
| gdr | 1 | 5.91927 | 15.780 | 15.850 | 1 |
| gdr | 2 | 5.95439 | 15.770 | 15.760 | 1 |
| gdr | 3 | 5.94739 | 15.700 | 15.740 | 1 |
| gdr | 4 | 5.93379 | 15.760 | 15.620 | 1 |
| gdr | 5 | 5.92167 | 15.840 | 15.630 | 1 |
| nogdr | 1 | 5.92959 | 16.400 | 16.390 | 0 |
| nogdr | 2 | 5.96725 | 16.810 | 16.770 | 0 |
| nogdr | 3 | 5.89401 | 16.470 | 16.320 | 0 |
| nogdr | 4 | 5.89517 | 16.320 | 16.370 | 0 |
| nogdr | 5 | 5.89246 | 16.420 | 16.300 | 0 |
