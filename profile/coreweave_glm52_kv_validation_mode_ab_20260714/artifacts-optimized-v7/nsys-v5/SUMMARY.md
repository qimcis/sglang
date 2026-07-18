# Nsight Systems Attribution

All modes use the optimized wheels and the same TP8 GLM-5.2 C1 8172-by-1000 workload. Nsight captured CUDA graph nodes and NCCL for 40 requested decode steps; profiler throughput is not used as an end-to-end performance result.

| Kernel | OFF count / avg us | Scheduler count / avg us | Fused count / avg us |
|---|---:|---:|---:|
| unprotected_topk | 6552 / 10.494 | 6552 / 10.495 | 0 / 0.000 |
| protected_topk_64 | 0 / 0.000 | 0 / 0.000 | 6552 / 12.743 |
| begin_status | 0 / 0.000 | 0 / 0.000 | 319 / 0.909 |
| failure_status | 0 / 0.000 | 0 / 0.000 | 312 / 1.718 |
| status_allreduce | 0 / 0.000 | 0 / 0.000 | 312 / 19.608 |
| checksum_scan | 8 / 350.008 | 8 / 349.940 | 8 / 350.284 |
| checksum_finalizer | 8 / 2.436 | 8 / 2.468 | 8 / 2.356 |

| Mode | GPU kernel instances | cudaLaunchKernel calls | cudaGraphLaunch calls | cudaMemcpyAsync calls |
|---|---:|---:|---:|---:|
| off | 1076827 | 9450 | 319 | 3639 |
| scheduler | 1085091 | 17714 | 319 | 4107 |
| fused | 1081130 | 13456 | 319 | 4672 |

- Begin-status kernels equal CUDA graph launches: 319/319.
- Final-status kernels equal status all-reduces: 312/312.
- Protected top-k runs 21 producers per completed status step.
- Protected top-k adds 47.205 us per status step versus scheduler top-k.
- Local begin/final status costs 2.627 us per status step.
- The TP status all-reduce costs 19.608 us per status step.
- Total directly attributed fused increment is 69.440 us per status step.
- Scheduler executes 12.696 more GPU kernels per status step than fused.
- Checksum scan and request-finalizer counts are identical in all modes (8 each).
