# Additional continuous batching validation results

## Summary by run

| Model tag | Resolution | Mode | Run | Status | Success | Throughput | Mean Lat | P99 | Peak Mem |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| flux_klein_small | 512x512 | continuous | 1 | PASS | 16/16 | 5.252 | 0.75 | 1.04 | 16690 |
| flux_klein_small | 512x512 | continuous | 2 | PASS | 16/16 | 6.410 | 0.62 | 0.71 | 18556 |
| flux_klein_small | 768x768 | continuous | 1 | PASS | 16/16 | 3.854 | 1.02 | 1.17 | 18556 |
| flux_klein_small | 768x768 | continuous | 2 | PASS | 16/16 | 3.957 | 0.99 | 1.07 | 18556 |
| flux_klein_small | 512x512 | no_batch | 1 | PASS | 16/16 | 6.127 | 0.59 | 0.68 | 16554 |
| flux_klein_small | 512x512 | no_batch | 2 | PASS | 16/16 | 6.273 | 0.58 | 0.69 | 16554 |
| flux_klein_small | 768x768 | no_batch | 1 | PASS | 16/16 | 3.507 | 1.05 | 1.52 | 17946 |
| flux_klein_small | 768x768 | no_batch | 2 | PASS | 16/16 | 3.920 | 0.92 | 1.04 | 17946 |
| flux_klein_base_small | 512x512 | continuous | 1 | PASS | 16/16 | 0.479 | 8.33 | 8.35 | 16748 |
| flux_klein_base_small | 512x512 | continuous | 2 | PASS | 16/16 | 0.473 | 8.44 | 8.53 | 18608 |
| flux_klein_base_small | 768x768 | continuous | 1 | PASS | 16/16 | 0.257 | 15.56 | 15.63 | 18608 |
| flux_klein_base_small | 768x768 | continuous | 2 | PASS | 16/16 | 0.257 | 15.55 | 15.61 | 18608 |
| flux_klein_base_small | 512x512 | no_batch | 1 | PASS | 16/16 | 0.447 | 8.06 | 9.35 | 16554 |
| flux_klein_base_small | 512x512 | no_batch | 2 | PASS | 16/16 | 0.453 | 8.00 | 8.87 | 16556 |
| flux_klein_base_small | 768x768 | no_batch | 1 | PASS | 16/16 | 0.253 | 14.32 | 15.86 | 17962 |
| flux_klein_base_small | 768x768 | no_batch | 2 | PASS | 16/16 | 0.253 | 14.31 | 15.82 | 17962 |
| ernie_turbo | 512x512 | continuous | 1 | PASS | 4/4 | 0.071 | 55.89 | 56.00 | 32436 |
| ernie_turbo | 512x512 | no_batch | 1 | PASS | 4/4 | 0.084 | 30.53 | 47.54 | 32056 |
| glm_image | 512x512 | continuous | 1 | FAIL | 0/4 | 0.000 | 0.00 | 0.00 | 0 |
| glm_image | 512x512 | no_batch | 1 | PASS | 4/4 | 0.069 | 36.38 | 57.73 | 36672 |
| ideogram4_fp8 | 512x512 | continuous | 1 | FAIL | 0/16 | 0.000 | 0.00 | 0.00 | 0 |
| ideogram4_fp8 | 512x512 | continuous | 2 | FAIL | 0/16 | 0.000 | 0.00 | 0.00 | 0 |
| ideogram4_fp8 | 768x768 | continuous | 1 | FAIL | 0/16 | 0.000 | 0.00 | 0.00 | 0 |
| ideogram4_fp8 | 768x768 | continuous | 2 | FAIL | 0/16 | 0.000 | 0.00 | 0.00 | 0 |
| ideogram4_fp8 | 512x512 | no_batch | 1 | FAIL | 0/16 | 0.000 | 0.00 | 0.00 | 0 |
| ideogram4_fp8 | 512x512 | no_batch | 2 | FAIL | 0/16 | 0.000 | 0.00 | 0.00 | 0 |
| ideogram4_fp8 | 768x768 | no_batch | 1 | FAIL | 0/16 | 0.000 | 0.00 | 0.00 | 0 |
| ideogram4_fp8 | 768x768 | no_batch | 2 | FAIL | 0/16 | 0.000 | 0.00 | 0.00 | 0 |
| sana600m512 | 512x512 | continuous | 1 | PASS | 4/4 | 1.317 | 3.03 | 3.04 | 7690 |
| sana600m512 | 512x512 | no_batch | 1 | PASS | 4/4 | 1.208 | 2.08 | 3.28 | 7692 |
| zimage | 512x512 | continuous | 1 | FAIL | 0/4 | 0.000 | 0.00 | 0.00 | 0 |
| zimage | 512x512 | no_batch | 1 | PASS | 4/4 | 0.373 | 6.70 | 10.64 | 20972 |

## Aggregates for completed matrices

| Model tag | Resolution | Mode | n | Throughput | Mean Lat | P99 | Peak Mem |
|---|---|---|---:|---:|---:|---:|---:|
| ernie_turbo | 512x512 | no_batch | 1 | 0.084 | 30.53 | 47.54 | 32056 |
| ernie_turbo | 512x512 | continuous | 1 | 0.071 | 55.89 | 56.00 | 32436 |
| flux_klein_base_small | 512x512 | no_batch | 2 | 0.450 | 8.03 | 9.11 | 16555 |
| flux_klein_base_small | 512x512 | continuous | 2 | 0.476 | 8.38 | 8.44 | 17678 |
| flux_klein_base_small | 768x768 | no_batch | 2 | 0.253 | 14.32 | 15.84 | 17962 |
| flux_klein_base_small | 768x768 | continuous | 2 | 0.257 | 15.56 | 15.62 | 18608 |
| flux_klein_small | 512x512 | no_batch | 2 | 6.200 | 0.59 | 0.69 | 16554 |
| flux_klein_small | 512x512 | continuous | 2 | 5.831 | 0.68 | 0.87 | 17623 |
| flux_klein_small | 768x768 | no_batch | 2 | 3.713 | 0.98 | 1.28 | 17946 |
| flux_klein_small | 768x768 | continuous | 2 | 3.905 | 1.01 | 1.12 | 18556 |
| glm_image | 512x512 | no_batch | 1 | 0.069 | 36.38 | 57.73 | 36672 |
| sana600m512 | 512x512 | no_batch | 1 | 1.208 | 2.08 | 3.28 | 7692 |
| sana600m512 | 512x512 | continuous | 1 | 1.317 | 3.03 | 3.04 | 7690 |
| zimage | 512x512 | no_batch | 1 | 0.373 | 6.70 | 10.64 | 20972 |

## Continuous vs no-batching where both passed

| Model tag | Resolution | Throughput Δ | Mean Lat Δ | P99 Δ |
|---|---|---:|---:|---:|
| ernie_turbo | 512x512 | -14.5% | +83.0% | +17.8% |
| flux_klein_base_small | 512x512 | +5.9% | +4.4% | -7.4% |
| flux_klein_base_small | 768x768 | +1.4% | +8.7% | -1.4% |
| flux_klein_small | 512x512 | -6.0% | +16.5% | +27.6% |
| flux_klein_small | 768x768 | +5.2% | +2.2% | -12.4% |
| sana600m512 | 512x512 | +9.0% | +46.0% | -7.5% |

## Failure notes

- `more_models/glm_image/continuous_512x512_inf_run1.json` failed with 0 completed and 4 failed requests.
- `more_models/ideogram4_fp8/continuous_512x512_inf_run1.json` failed with 0 completed and 16 failed requests.
- `more_models/ideogram4_fp8/continuous_512x512_inf_run2.json` failed with 0 completed and 16 failed requests.
- `more_models/ideogram4_fp8/continuous_768x768_inf_run1.json` failed with 0 completed and 16 failed requests.
- `more_models/ideogram4_fp8/continuous_768x768_inf_run2.json` failed with 0 completed and 16 failed requests.
- `more_models/ideogram4_fp8/no_batch_512x512_inf_run1.json` failed with 0 completed and 16 failed requests.
- `more_models/ideogram4_fp8/no_batch_512x512_inf_run2.json` failed with 0 completed and 16 failed requests.
- `more_models/ideogram4_fp8/no_batch_768x768_inf_run1.json` failed with 0 completed and 16 failed requests.
- `more_models/ideogram4_fp8/no_batch_768x768_inf_run2.json` failed with 0 completed and 16 failed requests.
- `more_models/zimage/continuous_512x512_inf_run1.json` failed with 0 completed and 4 failed requests.

Observed log failure patterns:


## Artifact layout

- Results: `results/`
- Logs: `logs/`
- Runner scripts: `run_*.sh`
