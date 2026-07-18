# Kimi K2.7 Direct-Kernel Attempt

Model: `moonshotai/Kimi-K2.7-Code`

Requested shape: one prefill + one decode on 16 total H100 GPUs.

Executed shape:

- Prefill pod: 8x H100, `--tp 8`
- Decode pod: 8x H100, `--tp 8`
- Transfer backend: `mooncake_tcp`
- Protection initially off for clean startup feasibility
- Direct checksum op built and verified in both pods before model startup

Capacity at start:

- Cluster allocatable: 240 GPUs across 30 ready GPU nodes
- Requested before Kimi pods: 224 GPUs
- Ready free full 8-GPU nodes: 2
- Kimi attempt consumed both free nodes, leaving 0 free GPUs while running

Result:

- Clean startup failed before serving.
- Prefill and decode both OOMed during model weight allocation.
- The direct checksum/page-tag fault matrix could not run because the model never reached a healthy server state.

Primary error:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 672.00 MiB.
GPU ... has a total capacity of 79.10 GiB of which 345.00 MiB is free.
Process ... has 78.75 GiB memory in use.
```

Retry:

- Retried with `--cpu-offload-gb 4` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Still OOMed during model initialization.

Artifacts:

- `prefill_clean.log`
- `decode_clean.log`
- `prefill_offload.log`
- `decode_offload.log`
- `PROTECTION_COVERAGE.md` (FA3 rerun requirements and Kimi K3 boundary)

Conclusion:

`moonshotai/Kimi-K2.7-Code` does not fit in this 1P1D 16xH100 total shape (`8xH100 prefill + 8xH100 decode`) with this runtime. A TP16-per-side run would require 32 H100 GPUs total for one prefill and one decode side, but only 16 GPUs were free.
