# Final continuous batching smoke validation

| Tag | Mode | File | Status | Success | Throughput | Mean Lat | P99 | Peak Mem |
|---|---|---|---|---:|---:|---:|---:|---:|
| flux_klein_small | continuous | `continuous_512x512_inf_run1.json` | PASS | 16/16 | 5.252 | 0.75 | 1.04 | 16690 |
| flux_klein_small | continuous | `continuous_512x512_inf_run2.json` | PASS | 16/16 | 6.410 | 0.62 | 0.71 | 18556 |
| flux_klein_small | continuous | `continuous_768x768_inf_run1.json` | PASS | 16/16 | 3.854 | 1.02 | 1.17 | 18556 |
| flux_klein_small | continuous | `continuous_768x768_inf_run2.json` | PASS | 16/16 | 3.957 | 0.99 | 1.07 | 18556 |
| flux_klein_small | no_batch | `no_batch_512x512_inf_run1.json` | PASS | 16/16 | 6.127 | 0.59 | 0.68 | 16554 |
| flux_klein_small | no_batch | `no_batch_512x512_inf_run2.json` | PASS | 16/16 | 6.273 | 0.58 | 0.69 | 16554 |
| flux_klein_small | no_batch | `no_batch_768x768_inf_run1.json` | PASS | 16/16 | 3.507 | 1.05 | 1.52 | 17946 |
| flux_klein_small | no_batch | `no_batch_768x768_inf_run2.json` | PASS | 16/16 | 3.920 | 0.92 | 1.04 | 17946 |
| flux_klein_base_small | continuous | `continuous_512x512_inf_run1.json` | PASS | 16/16 | 0.479 | 8.33 | 8.35 | 16748 |
| flux_klein_base_small | continuous | `continuous_512x512_inf_run2.json` | PASS | 16/16 | 0.473 | 8.44 | 8.53 | 18608 |
| flux_klein_base_small | continuous | `continuous_768x768_inf_run1.json` | PASS | 16/16 | 0.257 | 15.56 | 15.63 | 18608 |
| flux_klein_base_small | continuous | `continuous_768x768_inf_run2.json` | PASS | 16/16 | 0.257 | 15.55 | 15.61 | 18608 |
| flux_klein_base_small | no_batch | `no_batch_512x512_inf_run1.json` | PASS | 16/16 | 0.447 | 8.06 | 9.35 | 16554 |
| flux_klein_base_small | no_batch | `no_batch_512x512_inf_run2.json` | PASS | 16/16 | 0.453 | 8.00 | 8.87 | 16556 |
| flux_klein_base_small | no_batch | `no_batch_768x768_inf_run1.json` | PASS | 16/16 | 0.253 | 14.32 | 15.86 | 17962 |
| flux_klein_base_small | no_batch | `no_batch_768x768_inf_run2.json` | PASS | 16/16 | 0.253 | 14.31 | 15.82 | 17962 |
| ernie_turbo | continuous | `continuous_512x512_inf_run1.json` | PASS | 4/4 | 0.071 | 55.89 | 56.00 | 32436 |
| ernie_turbo | no_batch | `no_batch_512x512_inf_run1.json` | PASS | 4/4 | 0.084 | 30.53 | 47.54 | 32056 |
| glm_image | continuous | `continuous_512x512_inf_run1.json` | FAIL | 0/4 | 0.000 | 0.00 | 0.00 | 0 |
| glm_image | no_batch | `no_batch_512x512_inf_run1.json` | PASS | 4/4 | 0.069 | 36.38 | 57.73 | 36672 |
| ideogram4_fp8 | continuous | `continuous_512x512_inf_run1.json` | FAIL | 0/16 | 0.000 | 0.00 | 0.00 | 0 |
| ideogram4_fp8 | continuous | `continuous_512x512_inf_run2.json` | FAIL | 0/16 | 0.000 | 0.00 | 0.00 | 0 |
| ideogram4_fp8 | continuous | `continuous_768x768_inf_run1.json` | FAIL | 0/16 | 0.000 | 0.00 | 0.00 | 0 |
| ideogram4_fp8 | continuous | `continuous_768x768_inf_run2.json` | FAIL | 0/16 | 0.000 | 0.00 | 0.00 | 0 |
| ideogram4_fp8 | no_batch | `no_batch_512x512_inf_run1.json` | FAIL | 0/16 | 0.000 | 0.00 | 0.00 | 0 |
| ideogram4_fp8 | no_batch | `no_batch_512x512_inf_run2.json` | FAIL | 0/16 | 0.000 | 0.00 | 0.00 | 0 |
| ideogram4_fp8 | no_batch | `no_batch_768x768_inf_run1.json` | FAIL | 0/16 | 0.000 | 0.00 | 0.00 | 0 |
| ideogram4_fp8 | no_batch | `no_batch_768x768_inf_run2.json` | FAIL | 0/16 | 0.000 | 0.00 | 0.00 | 0 |
| sana600m512 | continuous | `continuous_512x512_inf_run1.json` | PASS | 4/4 | 1.317 | 3.03 | 3.04 | 7690 |
| sana600m512 | no_batch | `no_batch_512x512_inf_run1.json` | PASS | 4/4 | 1.208 | 2.08 | 3.28 | 7692 |
| zimage | continuous | `continuous_512x512_inf_run1.json` | FAIL | 0/4 | 0.000 | 0.00 | 0.00 | 0 |
| zimage | no_batch | `no_batch_512x512_inf_run1.json` | PASS | 4/4 | 0.373 | 6.70 | 10.64 | 20972 |
| glm_image_retry | continuous | `glm_image_continuous_512x512_smoke.json` | PASS | 4/4 | 0.068 | 58.71 | 58.80 | 36676 |
| glm_image_retry | no_batch | `glm_image_no_batch_512x512_smoke.json` | PASS | 4/4 | 0.069 | 36.24 | 57.61 | 36670 |
| ideogram4_fp8_retry | continuous | `ideogram4_fp8_continuous_512x512_smoke.json` | FAIL | 0/4 | 0.000 | 0.00 | 0.00 | 0 |
| ideogram4_fp8_retry | continuous | `ideogram4_fp8_continuous_retry_512x512_smoke.json` | PASS | 4/4 | 0.245 | 16.00 | 16.32 | 29122 |
| ideogram4_fp8_retry | no_batch | `ideogram4_fp8_no_batch_512x512_smoke.json` | PASS | 4/4 | 0.225 | 11.98 | 17.63 | 28072 |
| zimage_retry | continuous | `zimage_continuous_512x512_smoke.json` | PASS | 4/4 | 0.295 | 13.48 | 13.56 | 21262 |
| zimage_retry | no_batch | `zimage_no_batch_512x512_smoke.json` | PASS | 4/4 | 0.371 | 6.74 | 10.71 | 20972 |

## Fix verification

- Z-Image continuous retry passed after fixing packed list merge for per-sample captions.
- GLM continuous retry passed after cloning scheduler runtime per request in continuous mode.
- Ideogram no-batch and continuous retry passed after adding CUDA 13 NVRTC builtins to `LD_LIBRARY_PATH` and disabling generic packed denoising for custom `_run_denoising_step` subclasses.
- Remote GPU was freed after the runs (`0 MiB` used).
