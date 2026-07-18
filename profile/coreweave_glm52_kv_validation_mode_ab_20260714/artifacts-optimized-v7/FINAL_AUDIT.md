# Final KV Protection Optimization Audit

## Outcome

All six requested optimizations are implemented and the acceptance criteria pass on the final optimized wheels. The same counterbalanced GLM-5.2 H200 TP8 PD workload completed 690 measured requests with zero request failures and zero drain timeouts. The final Nsight campaign captured OFF, Scheduler, and Fused attribution with the optimized wheels and restored the original Helm state afterward.

Final wheels:

- `sglang_kernel-0.4.4-cp310-abi3-linux_x86_64.whl`: `a2dc93d82b964175ff39ce24ab67e2a84897a6de1b69de5f51af1b71415e142e`
- `sglang-0.0.0.dev1+g0e8b11e45.kvexact20260715-cp312-cp312-linux_x86_64.whl`: `7afe5c09e771abd4717f87c3af10453d1ca5fa6b7e1680e29d1ec5c8325da0f8`

## Acceptance Criteria

### 1. Fused per-step status orchestration

Pass.

- `kv_page_protection_begin_forward_kernel` clears status and advances epochs in one graph-safe CUDA operation.
- `kv_page_protection_failure_status_kernel` emits status and failure tensors in one graph-safe CUDA operation.
- Focused tests cover request slot 0, negative and out-of-range indices, status bits, incomplete publication, eager execution, and CUDA graph replay.
- Nsight captured 319 begin kernels for 319 CUDA graph launches and 312 final-status kernels for 312 status all-reduces.
- Scheduler executed 12.696 more GPU kernels per completed status step than Fused, confirming removal of the prior small-op orchestration.

### 2. Once-per-epoch publication and page-size-64 top-k

Pass.

- Completion publication uses a conditional `atomicCAS`; only the first producer can publish an epoch. All producers still validate and accumulate status.
- Page size 64 uses shift/mask decomposition while other page sizes retain the generic division/remainder path.
- Final H200 protected top-k coverage passed 30 tests across naive/radix behavior, every status bit, malformed slots and lengths, graph padding, replay, generic page size, page size 64, sanitization, and revalidation after publication.
- Compute Sanitizer passed the 30 protected top-k tests with `ERROR SUMMARY: 0 errors`.
- Final SM90a SASS inspection found 11 `MUFU.RCP` instructions in the generic specialization, zero in the page-size-64 specialization, and 12 immediate shift-by-6 instructions in the page-size-64 specialization.
- Nsight captured 6,552 page-size-64 protected top-k kernels only in Fused. They averaged 12.743 us versus 10.495 us for unprotected Scheduler top-k, or 47.205 us incremental cost across 21 producers per status step.

### 3. Compact checksum roots and materialization

Pass.

- The compact page-enabled scan no longer atomically fans request-root contributions into the root accumulator. The request finalizer XOR-reduces low 32-bit raw page accumulators to derive the exact legacy root.
- One request-oriented finalizer writes roots and compact page digests using prefix offsets. The legacy dense operator and schema remain unchanged.
- The manager allocates exactly `batch_size + sum(page_counts)` result values and performs one packed D2H materialization. The production result path has no GPU `torch.cat`, root/page result clone, dense padded page finalization, or second result D2H.
- Focused checksum coverage passed 24 tests for dense ABI compatibility, exact root/page parity, empty and unaligned ranges, int32/int64 tables, corruption localization, ragged batches, workspace growth and reuse, multiple outstanding batches, and SWA two-pass behavior.
- Compute Sanitizer passed all 24 checksum tests with `ERROR SUMMARY: 0 errors`.
- Nsight captured one request finalizer per TP rank: 8 scan and 8 finalizer kernels in each mode. Finalizer averages were 2.436 us OFF, 2.468 us Scheduler, and 2.356 us Fused.

### 4. Shipped-path correctness and memory safety

Pass.

- The final combined CUDA and manager suite passed 178 tests.
- The final manager/retry subset passed 12 tests, including TP remote failure propagation, deferred materialization, survivor continuity, release ordering, and pending collective drain behavior.
- Protected top-k and checksum Compute Sanitizer runs reported zero memory errors.
- Failed selected entries are sanitized to allocator-reserved slot 0; request slot 0 remains graph padding rather than a real request.

### 5. Same-workload performance and attribution

Pass.

- Sequence: `OFF1, Scheduler1, Fused1, Fused2, Scheduler2, OFF2`.
- Model/request shape: GLM-5.2 FP8, TP8 PD, 8,172 input tokens, 1,000 output tokens, 30-second warmup, 120-second measurements, C1/C2/C4/C8.
- Nodes remained fixed at prefill `ge53750` and decode `g126cd6`; transfer checksum was enabled in every arm.
- Every retained decode pod used the final wheels, the expected mode flags, the same image and decode node, and had zero container restarts.
- The first OFF1 measurement was rejected after a proxy restart and overwritten. All six retained samples independently validate.

| C | Scheduler/OFF throughput | Fused/OFF throughput | Scheduler/OFF TPOT | Fused/OFF TPOT | Fused/OFF TTFT |
|---:|---:|---:|---:|---:|---:|
| 1 | +0.463% | -0.624% | -0.501% | +0.621% | -0.083% |
| 2 | -0.214% | -0.936% | +0.132% | +0.919% | -0.093% |
| 4 | -0.173% | -0.877% | +0.141% | +0.753% | -0.046% |
| 8 | -0.472% | -0.714% | +0.322% | +0.465% | +0.182% |

Nsight directly attributes 69.440 us per completed Fused status step:

- Protected top-k increment: 47.205 us.
- Local begin/final status kernels: 2.627 us.
- TP status all-reduce: 19.608 us.

The profiler workload is attribution evidence only; the counterbalanced campaign above is the end-to-end performance result.

### 6. Deferred work and source scope

Pass.

- No packed sidecar representation was added.
- Protected top-k block size was not changed.
- No warp-level deduplication was added.
- The scoped optimization source diff passes `git diff --check`.
- Repository-wide whitespace failures and unrelated FA3, multimodal, benchmark, credential, and user artifact changes predate or are outside this work and were not modified as part of the optimization.
- Independent final review found no functional issue in the scoped diff.

## Evidence

- End-to-end results: `SUMMARY.md` and `summary.json` in this directory.
- Nsight attribution: `nsys-v5/SUMMARY.md`, `nsys-v5/summary.json`, the three CSV exports, and the three `.nsys-rep` files.
- Retained workload inputs and pod snapshots: `{off,scheduler,fused}/repeat{1,2}.json` and corresponding decode logs/pod JSON.
- Reproducible SM90a inspection: `../audit_topk_sass.sh`.
- Reproducible analyzers: `../analyze.py` and `../analyze_nsys.py`.

## Residual Risk

- Runtime and sanitizer validation is H200/SM90-specific; other CUDA architectures were not exercised in this campaign.
- Compact offsets are produced by the shipped manager and kernel writes are bounds-guarded, but the low-level compact operator expects semantically valid caller-provided prefix offsets.
- Publication ordering relies on the shipped same-stream producer sequence; concurrent cross-stream duplicate request producers are outside the supported execution path.
