# GLM-5.2 KV Validation Mode A/B

Artifact set: `artifacts-optimized-v7`.
All 690 measured requests completed with 0 failures and 0 drain timeouts.
Transfer checksum is enabled identically in all three arms.

| C | OFF tok/s | Scheduler tok/s | Fused tok/s | Scheduler/OFF | Fused/OFF | Fused/Scheduler |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 73.35 | 73.69 | 72.90 | +0.463% | -0.624% | -1.082% |
| 2 | 139.56 | 139.27 | 138.26 | -0.214% | -0.936% | -0.724% |
| 4 | 257.51 | 257.06 | 255.25 | -0.173% | -0.877% | -0.705% |
| 8 | 447.83 | 445.72 | 444.64 | -0.472% | -0.714% | -0.243% |

| C | OFF TPOT ms | Scheduler TPOT ms | Fused TPOT ms | Scheduler/OFF | Fused/OFF |
|---:|---:|---:|---:|---:|---:|
| 1 | 13.362 | 13.295 | 13.445 | -0.501% | +0.621% |
| 2 | 13.886 | 13.904 | 14.013 | +0.132% | +0.919% |
| 4 | 15.081 | 15.102 | 15.194 | +0.141% | +0.753% |
| 8 | 17.419 | 17.475 | 17.500 | +0.322% | +0.465% |

| C | OFF TTFT ms | Scheduler TTFT ms | Fused TTFT ms | Scheduler/OFF | Fused/OFF |
|---:|---:|---:|---:|---:|---:|
| 1 | 40.631 | 40.382 | 40.598 | -0.612% | -0.083% |
| 2 | 40.597 | 40.515 | 40.559 | -0.201% | -0.093% |
| 4 | 40.496 | 40.557 | 40.478 | +0.150% | -0.046% |
| 8 | 40.494 | 40.495 | 40.567 | +0.004% | +0.182% |
