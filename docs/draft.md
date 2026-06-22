# Draft Plan

## Task Contract

- Task name: LingBot fused residual-camera-layernorm chain.
- Objective: Replace the causal LingBot post-self-attention chain `residual + gate * attn_output -> camera affine -> layernorm` with a fused implementation that avoids computing the currently unused pre-camera norm and reduces memory traffic in `CausalLingBotWorldTransformerBlock.forward`.
- Correctness requirements: Preserve outputs of the existing LingBot causal block chain within normal FP32/BF16 tolerance; preserve `gate_msa` framewise broadcasting semantics; preserve behavior when camera conditioning is absent; preserve existing non-CUDA/unsupported-shape fallback behavior; do not alter causal KV-cache, cross-attention cache, sequence-sharding, or denoising semantics.
- Performance or quality target: Reduce per-block post-self-attention overhead by eliminating one redundant layernorm and fusing residual, camera affine, and final layernorm work for CUDA-supported LingBot hidden size (`D=5120`, multiple of 256). Expected end-to-end chunk reward is modest but meaningful, roughly low single-digit percent unless profiling shows layernorm/elementwise work dominates.
- Allowed implementation approaches: Small SGLang-local Python/CuTe DSL integration; prefer extending existing `sglang.jit_kernel.diffusion.cutedsl.scale_residual_norm_scale_shift` patterns; keep a simple native fallback for CPU/ROCm/unsupported shapes; no changes under `/Users/chimcisaac/kernel-design-agents`.
- Validation command: `python3 -m py_compile python/sglang/multimodal_gen/runtime/models/dits/lingbot_world.py python/sglang/multimodal_gen/runtime/layers/layernorm.py python/sglang/jit_kernel/diffusion/cutedsl/scale_residual_norm_scale_shift.py python/sglang/multimodal_gen/test/unit/realtime/test_lingbot_causal_denoising.py`; `python3 -m pytest python/sglang/multimodal_gen/test/unit/realtime/test_lingbot_causal_denoising.py -q` when pytest is available; `git diff --check`.
- Evaluation command: If CUDA dependencies are available, run or add a targeted microbenchmark comparing old split chain vs fused LingBot chain at LingBot shapes (`B=1`, `F=3`, `D=5120`, representative sequence length). In this local environment CUDA/pytest are not available, so record validation-only evidence if blocked.
- Promotion criteria: Candidate is promoted only if the native reference path matches the previous chain in unit tests, syntax/whitespace checks pass, the CUDA path is guarded by the same hidden-dim constraints as existing fused norm kernels, and unsupported platforms/shapes fall back safely.

## Baseline

- Current implementation: `CausalLingBotWorldTransformerBlock.forward` calls `self.self_attn_residual_norm(hidden_states, attn_output, gate_msa, zero, zero)`, which computes and returns `norm(residual + gate * attn_output)` even though LingBot then applies `cam_conditioner` to the residual output and overwrites `norm_hidden_states` with `self.self_attn_residual_norm.norm(hidden_states)`. This creates an unnecessary pre-camera norm plus extra memory reads/writes.
- Existing tests or validators: `python/sglang/multimodal_gen/test/unit/realtime/test_lingbot_causal_denoising.py` covers LingBot camera conditioner caching and causal denoising helpers. Existing fused norm kernel tests are in `python/sglang/jit_kernel/tests/diffusion/test_fused_norm_scale_shift.py`.
- Baseline validation result: `python3 -m py_compile python/sglang/multimodal_gen/runtime/models/dits/lingbot_world.py python/sglang/multimodal_gen/runtime/layers/layernorm.py` passed. `python3 -m pytest python/sglang/multimodal_gen/test/unit/realtime/test_lingbot_causal_denoising.py -q` failed because local Python lacks `pytest`.
- Baseline performance result: Not measured; no CUDA benchmarking was run in this local environment.

## Risks And Unknowns

- The existing fused norm kernel supports CUDA CuTe DSL only for `D % 256 == 0` and `D <= 8192`; LingBot `D=5120` satisfies this, but tests in this environment cannot exercise CUDA.
- LayerNorm numerical parity must match the existing FP32 LayerNorm path, including optional affine weight/bias from `self.self_attn_residual_norm.norm`.
- `gate_msa`, `cam_scale`, and `cam_shift` can be frame-shaped tensors (`[B, F, 1, D]`) and must broadcast over `S = F * tokens_per_frame` exactly like existing helpers.
- Camera conditioning may be absent (`c2ws_plucker_emb is None`), in which case the fused helper must behave like residual + final norm without applying camera affine.
- Adding a new custom op increases import surface for environments without CUDA/CuTe dependencies; imports must remain lazy behind CUDA paths as in the existing layernorm kernels.

## Candidate Directions

1. Candidate name: native-helper-first. Add a LingBot-specific helper method that computes residual, optional camera affine, and final norm once using existing PyTorch/native ops, then route the causal block through it. Expected value: removes the redundant first norm in all environments and is easy to validate. Risk: may be slower on CUDA than the existing fused residual kernel if no custom CUDA op is active.
2. Candidate name: CuTe fused residual-camera-layernorm op. Extend the existing CuTe DSL fused residual/norm kernel module with a new custom op that computes residual, optional camera affine, and layernorm in one CTA per `(B,S)` row, returning both camera-conditioned hidden state and normalized hidden state. Expected value: intended kernel fusion with reduced memory traffic. Risk: CUDA path cannot be locally runtime-tested here.

## First Implementation Steps

1. Add unit-testable reference coverage for the LingBot residual-camera-layernorm chain using a small fake block/norm.
2. Add a native helper on `CausalLingBotWorldTransformerBlock` that computes the fused semantics once and use it in `forward`.
3. If practical within the existing kernel patterns, add a lazy CUDA custom-op wrapper in `layernorm.py`/CuTe DSL and let the helper use it when supported.
4. Validate syntax and run available tests; record any environment blockers.

## Evidence Required For Promotion

- Correctness evidence: A targeted LingBot unit test compares the new helper against the previous split chain with and without camera scale/shift; syntax compilation passes.
- Performance or quality evidence: CUDA microbenchmark when available, or a documented blocker if local CUDA/pytest dependencies are absent.
- Regression checks: Existing LingBot denoising unit test file remains import/compile valid; `git diff --check` passes; no cache/sequence-sharding code paths are changed.

---

# Draft Plan: Ideogram4 Fused FP8 GEMM

## Task Contract

- Task name: Ideogram4 weight-only FP8 fused GEMM wiring.
- Objective: Replace the current eager row-wise FP8 weight dequantization before every linear with existing fused FP8 GEMM kernels where supported, especially for Ideogram4 DiT and Qwen3-VL text encoder weight-only FP8 modules.
- Correctness requirements: Preserve checkpoint loading from `[out, in]` FP8 weights plus row-wise `weight_scale`; preserve CPU/meta/non-CUDA fallback behavior; preserve tensor-parallel column sharding and gather semantics; preserve output dtype and bias behavior; avoid changing NVFP4 or ModelOpt FP8 quantized paths.
- Performance or quality target: Remove full-weight dequantization from the hot forward path on CUDA devices supported by SGLang FP8 GEMM kernels. Expected win is shape/hardware dependent, with the largest value on Ideogram4 FP8 denoising linears and text encoder linears.
- Allowed implementation approaches: Reuse existing SGLang fused FP8 utilities such as `sglang.srt.layers.quantization.fp8_utils.apply_fp8_linear`; keep changes localized to diffusion weight-only FP8 wrappers when possible; do not add a new CUDA kernel unless no existing fused path fits; do not work under `/Users/chimcisaac/kernel-design-agents`.
- Validation command: `python3 -m py_compile python/sglang/multimodal_gen/runtime/layers/quantization/weight_only_fp8.py python/sglang/multimodal_gen/runtime/models/dits/ideogram.py python/sglang/multimodal_gen/runtime/models/encoders/qwen3vl.py python/sglang/multimodal_gen/test/unit/test_ideogram4.py`; `python3 -m pytest python/sglang/multimodal_gen/test/unit/test_ideogram4.py -q` when pytest is available; `git diff --check`.
- Evaluation command: On a CUDA host with Ideogram4 FP8 available, run an Ideogram4 `sglang generate` perf dump before/after and compare denoising latency; optionally profile to confirm `dequantize_rowwise_fp8_weight` no longer appears in CUDA hot path and FP8 GEMM kernels are used.
- Promotion criteria: Promote only if CPU/meta tests still pass, CUDA-supported modules use fused `apply_fp8_linear` without per-forward full weight dequantization, unsupported platforms safely fall back to the previous eager dequantization path, and no non-FP8 quantized paths regress.

## Baseline

- Current implementation: `WeightOnlyFP8Linear.forward` and `WeightOnlyFP8ColumnParallelLinear.forward` dequantize FP8 weights to compute dtype on every call, then call `F.linear`. This is in `python/sglang/multimodal_gen/runtime/layers/quantization/weight_only_fp8.py`.
- Existing fused path: `ModelOptFp8LinearMethod` and `Fp8LinearMethod` call `apply_fp8_linear`, which can dispatch to `sgl_kernel::fp8_scaled_mm`, Triton, or torch scaled-mm paths depending on scale layout and hardware.
- Existing tests or validators: `python/sglang/multimodal_gen/test/unit/test_ideogram4.py` covers row-wise dequantization, weight-only FP8 loading, Ideogram DiT TP linears, and text encoder FP8 wrappers.
- Baseline validation result: Not rerun yet for this task.
- Baseline performance result: Not measured locally; this machine is not assumed to have the target CUDA/B200 runtime.

## Risks And Unknowns

- `apply_fp8_linear` expects transposed weight layout `[in, out]`, while checkpoint loading currently stores `[out, in]`; the fused path must pass a transpose view without changing loader-facing parameter layout.
- CUTLASS fused path requires compatible shapes and CUDA support; unsupported shapes must fall back safely.
- Existing tests may assert current weight parameter shapes after meta construction, so layout conversion should not break pre-load/meta expectations.
- Tensor-parallel column shards have local `out_features_per_partition`; scale and weight layout must remain aligned after sharding.
- Text encoder loading uses a custom `load_weights` path rather than DiT FSDP loader hooks, so a global post-load hook may not cover all modules.

## Candidate Directions

1. Candidate name: lazy-fused-weight-only-fp8. Keep checkpoint parameters in load-friendly `[out, in]` layout, pass a non-materializing transposed `[in, out]` FP8 view on CUDA fused forward, and call `apply_fp8_linear`; fallback to current eager dequantization on unsupported devices. Expected value: minimal integration risk and covers DiT/text encoder modules uniformly. Risk: fused kernel layout assumptions must match the existing ModelOpt FP8 path.
2. Candidate name: post-load-transpose-weight-only-fp8. Add an explicit post-load conversion helper and call it from Ideogram DiT/text encoder load paths. Expected value: no first-call layout conversion. Risk: more invasive loader hooks and higher chance of missing text encoder/offload paths.

## First Implementation Steps

1. Add a fused-forward helper in `weight_only_fp8.py` that detects CUDA-compatible FP8 weights and calls `apply_fp8_linear` with a transposed weight view.
2. Preserve current eager dequantization fallback and expose a small method or flag that tests can inspect.
3. Update unit tests to verify CPU fallback parity and fused-path dispatch behavior under mocks without requiring CUDA.
4. Run syntax, unit tests if available, and whitespace validation.

## Evidence Required For Promotion

- Correctness evidence: Existing Ideogram FP8 unit tests pass, plus a targeted test validates fallback parity and mocked fused dispatch.
- Performance or quality evidence: Local evidence should show the fused path calls `apply_fp8_linear` instead of `dequantize_rowwise_fp8_weight`; GPU perf numbers are required before claiming speedup.
- Regression checks: `git diff --check`; no changes to NVFP4, ModelOpt FP8, or unrelated quantization behavior.

---

# Draft Plan: Fused CUDA KV Transfer Checksum

## Task Contract

- Task name: Fused CUDA KV transfer checksum.
- Objective: Replace the current Python/Torch checksum path that materializes logical KV rows and mixes lanes from Python with a fused CUDA kernel exposed through `sgl-kernel`, then use it from `python/sglang/srt/mem_cache/kv_page_tags.py` when rows are CUDA tensors.
- Correctness requirements: Match the existing `hash_rows_with_positions` bit-for-bit for contiguous logical-row tensors, selected logical token indices, optional lane cap, and position mixing; never hash physical page ids; preserve CPU/non-CUDA fallback; preserve signed int64 bit-pattern handling for metadata transport.
- Performance or quality target: Reduce `always_full` checksum overhead by avoiding temporary selected row materialization plus Python-level lane mixing; expected improvement should be visible in the standalone CUDA checksum microbenchmark.
- Allowed implementation approaches: CUDA/C++ custom op in `sgl-kernel` only (no Triton); Python wrapper in `sgl_kernel.kvcacheio`; optional import/fallback in SGLang runtime; no changes to KDA reference repo.
- Validation command: `python3 -m py_compile python/sglang/srt/mem_cache/kv_page_tags.py sgl-kernel/python/sgl_kernel/kvcacheio.py sgl-kernel/tests/test_kv_checksum.py`; CUDA validation with `python3 -m pytest sgl-kernel/tests/test_kv_checksum.py -q` in an environment where `sgl-kernel` can be rebuilt/installed.
- Evaluation command: CUDA microbenchmark comparing existing `hash_rows_with_positions` vs new `sgl_kernel.kvcacheio.kv_checksum` for representative `(num_tokens,row_bytes)` shapes.
- Promotion criteria: Candidate is promoted if CUDA op output matches current Python/Torch checksum across full and partial lane modes, fallback still works without `sgl_kernel`, and benchmark shows lower latency or at least no regression for realistic rows.

## Baseline

- Current implementation: `compute_transfer_checksum` gathers logical KV rows, `hash_kv_rows` selects sampled rows, and `hash_rows_with_positions` loops over int64 lanes from Python while doing vectorized Torch mixing and a final XOR reduce. This causes many kernel launches and materializes intermediate row tensors.
- Existing tests or validators: `test/registered/unit/disaggregation/test_kv_transfer_checksums.py` checks logical-order and physical-id independence. No sgl-kernel op exists yet.
- Baseline validation result: Focused unit tests passed in the H100 staging container before this kernel task (`39 passed`).
- Baseline performance result: Full ON PD serving benchmark was much slower than OFF; standalone helper benchmark showed `always_full` checksum taking tens to hundreds of ms depending on row size.

## Risks And Unknowns

- `sgl-kernel` build can be expensive and may need a CUDA Linux environment; local macOS can only syntax-check Python files.
- Exact bit-for-bit parity requires matching splitmix64 and signed int64 bit-pattern semantics.
- `rows.view(torch.uint8)` may represent arbitrary input dtype; the CUDA op will require a contiguous byte view and a row byte count divisible by/padded to int64 lanes at the wrapper boundary.
- The first candidate operates on already-gathered logical rows. A later candidate should hash directly from KV cache layer buffers and `kv_loc`, avoiding `gather_logical_kv_rows` entirely.

## Candidate Directions

1. Candidate name: contiguous-row CUDA checksum. Add `kv_checksum(Tensor rows_uint8, Tensor token_indices, int num_lanes, bool include_positions) -> Tensor` and use it when `rows` is CUDA. Expected value: removes Python lane loop and row selection materialization. Risk: still requires `gather_logical_kv_rows` before hashing.
2. Candidate name: direct-KV-location CUDA checksum. Add a kernel that reads K/V layer buffers by `kv_loc` and hashes directly. Expected value: largest improvement. Risk: larger integration surface across KV layouts and layer accessors.

## First Implementation Steps

1. Implement candidate-001 contiguous-row CUDA checksum in `sgl-kernel/csrc/kvcacheio/checksum.cu`.
2. Register op in `common_extension.cc`, add prototype to `sgl_kernel_ops.h`, add source to CMake, and expose Python wrapper in `sgl_kernel.kvcacheio`.
3. Add `sgl-kernel/tests/test_kv_checksum.py` parity and benchmark smoke tests.
4. Modify `kv_page_tags.hash_rows_with_positions` to call the CUDA op for CUDA contiguous rows with fallback on import/op failure.
5. Validate syntax locally and record candidate evidence.

## Evidence Required For Promotion

- Correctness evidence: `sgl-kernel/tests/test_kv_checksum.py` parity passes on CUDA; existing KV protection tests still pass.
- Performance or quality evidence: microbenchmark shows lower checksum time for at least 4MB/16MB/67MB logical rows.
- Regression checks: CPU fallback unchanged; `git diff --check` and Python compile checks pass.
