# GDR root-cause sweep summary

- trials per config: 3
- metrics from nccl-tests all_reduce_perf logs

| case | avg BW mean | 1GiB out mean | 1GiB in mean | GDR markers | coll channels |
|---|---:|---:|---:|---|---|
| bind_core_gdr | 5.91830 | 15.777 | 15.800 | [1, 1, 1] | [2, 2, 2] |
| bind_core_nogdr | 5.88659 | 16.320 | 16.397 | [0, 0, 0] | [2, 2, 2] |
| bind_none_gdr | 6.03492 | 15.750 | 15.777 | [1, 1, 1] | [2, 2, 2] |
| bind_none_nogdr | 5.84532 | 16.083 | 16.127 | [0, 0, 0] | [2, 2, 2] |
| bind_numa_gdr | 6.04009 | 15.687 | 15.760 | [1, 1, 1] | [2, 2, 2] |
| bind_numa_nogdr | 5.92251 | 16.177 | 16.203 | [0, 0, 0] | [2, 2, 2] |
| gdrread0_gdr | 3.31500 | 9.333 | 9.343 | [0, 0, 0] | [2, 2, 2] |
| gdrread1_gdr | 5.99184 | 15.720 | 15.820 | [1, 1, 1] | [2, 2, 2] |
| qps1_gdr | 6.02596 | 15.770 | 15.807 | [1, 1, 1] | [2, 2, 2] |
| qps2_gdr | 5.30567 | 13.307 | 13.273 | [1, 1, 1] | [2, 2, 2] |
| qps4_gdr | 5.28848 | 13.097 | 13.087 | [1, 1, 1] | [2, 2, 2] |
