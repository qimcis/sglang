# Implementation Plan

## Candidate 1: LingBot Residual-Camera-Norm Helper

1. Add a method on `CausalLingBotWorldTransformerBlock` that implements the intended fused semantics:
   - `residual = hidden_states + gate_msa * attn_output`
   - if camera conditioning exists: `hidden = (1 + cam_scale) * residual + cam_shift`
   - `norm_hidden = self.self_attn_residual_norm.norm(hidden).to(orig_dtype)`
   - return `(norm_hidden, hidden)`.
2. Preserve old behavior through a test reference rather than keeping the redundant pre-camera norm call in production.
3. Add a targeted unit test to `test_lingbot_causal_denoising.py` that verifies the helper matches the old split chain for framewise gates and supplied camera scale/shift.
4. Replace lines 1054-1071 in `CausalLingBotWorldTransformerBlock.forward` with the helper.
5. Record candidate evidence in `candidates.jsonl` without overwriting prior candidate history.

## Candidate 2: CuTe DSL Residual-Camera-LayerNorm Kernel

1. Extend `python/sglang/jit_kernel/diffusion/cutedsl/scale_residual_norm_scale_shift.py` with `sglang::fused_lingbot_residual_cam_layernorm`.
2. Compute `residual + gate * x`, optional camera scale/shift, store camera-conditioned hidden, then layernorm that value in one CTA per `(B, S)` row.
3. Route `ScaleResidualLayerNormScaleShift.forward_lingbot_residual_cam_norm` to the fused op only on CUDA for supported layernorm shapes (`D % 256 == 0`, `D <= 8192`), with native fallback elsewhere.
4. Let the LingBot block helper call this hook when present.
5. Record validation and any local runtime blockers in `candidates.jsonl`.

## Validation

- `python3 -m py_compile python/sglang/multimodal_gen/runtime/models/dits/lingbot_world.py python/sglang/multimodal_gen/runtime/layers/layernorm.py python/sglang/jit_kernel/diffusion/cutedsl/scale_residual_norm_scale_shift.py python/sglang/multimodal_gen/test/unit/realtime/test_lingbot_causal_denoising.py`
- `python3 -m pytest python/sglang/multimodal_gen/test/unit/realtime/test_lingbot_causal_denoising.py -q` when pytest is installed
- `git diff --check`

## Promotion

- Promote Candidate 1 if the targeted reference test is added, syntax validation passes, and the only runtime-test blocker is missing local pytest/CUDA dependencies.
- Promote Candidate 2 as accepted-pending-runtime-validation only if syntax/whitespace pass locally and CUDA parity/performance is explicitly left to CI or a GPU host because this local environment lacks runtime dependencies.

---

# Implementation Plan: Ideogram4 Fused FP8 GEMM

## Candidate 1: Lazy Fused Weight-Only FP8 Forward

1. Extend `WeightOnlyFP8Linear` and `WeightOnlyFP8ColumnParallelLinear` with a private fused-forward attempt.
2. Keep the public checkpoint parameter layout unchanged as `[out, in]` so current loaders and meta-shape tests remain valid.
3. On CUDA/non-meta forward, pass a non-materializing transposed FP8 weight view in `[in, out]` layout expected by `apply_fp8_linear`.
4. Call `apply_fp8_linear(input=x, weight=transposed_weight, weight_scale=weight_scale, input_scale=None, bias=bias, cutlass_fp8_supported=cutlass_fp8_supported())`.
5. Fall back to current `dequantize_rowwise_fp8_weight + F.linear` for CPU/meta/non-CUDA, import/runtime failures, or unsupported kernel conditions.
6. Preserve TP gather behavior by applying fused GEMM to the local column shard, then using existing `tensor_model_parallel_all_gather` when `gather_output` is true.

## Candidate 2: Post-Load Layout Conversion

1. If Candidate 1 is correct but the transpose-view layout is incompatible with a target fused backend, add explicit post-load conversion helpers for loaded Ideogram DiT and text encoder modules.
2. Wire those helpers through `Ideogram4Transformer2DModel.post_load_weights()` and `IdeogramQwen3VLTextEncoder.load_weights()`.
3. Promote only if loader coverage is complete for FSDP, component offload, and text encoder loading.

## Validation

- `python3 -m py_compile python/sglang/multimodal_gen/runtime/layers/quantization/weight_only_fp8.py python/sglang/multimodal_gen/runtime/models/dits/ideogram.py python/sglang/multimodal_gen/runtime/models/encoders/qwen3vl.py python/sglang/multimodal_gen/test/unit/test_ideogram4.py`
- `python3 -m pytest python/sglang/multimodal_gen/test/unit/test_ideogram4.py -q` when pytest is installed
- `git diff --check`

## Evaluation

- On CUDA target hardware, compare Ideogram4 FP8 denoising perf dumps before/after with `--backend sglang --enable-torch-compile --warmup`.
- Use profiler/NVTX to confirm fused FP8 GEMM kernels replace per-forward full-weight dequantization in hot linears.

## Promotion

- Promote Candidate 1 if validation passes locally and mocked tests prove dispatch to `apply_fp8_linear` with safe fallback.
- Require GPU benchmark evidence before claiming measured performance improvement.

---

# Implementation Plan: Fused CUDA KV Transfer Checksum

## Candidate 1: Contiguous Logical-Row CUDA Checksum

1. Add `csrc/kvcacheio/checksum.cu` implementing `kv_checksum(Tensor rows, Tensor token_indices, int num_lanes, bool include_positions) -> Tensor`.
2. The kernel hashes selected rows in logical token-index order using the same splitmix64 bit-pattern math as `kv_page_tags.hash_rows_with_positions`.
3. Avoid materializing `rows.index_select(0, token_indices)` inside Python. The kernel reads selected rows directly from `rows` and reduces to one int64 digest tensor.
4. Register the op in `common_extension.cc`, declare it in `include/sgl_kernel_ops.h`, add the source to `CMakeLists.txt`, and expose a Python wrapper in `sgl_kernel.kvcacheio`.
5. Wire `python/sglang/srt/mem_cache/kv_page_tags.py` to use the op for CUDA tensors, with fallback to the current Torch implementation for CPU, unsupported inputs, or import/runtime failures.

## Candidate 2: Direct KV-Location Checksum

1. Later optimization: hash directly from KV cache layer buffers and `kv_loc`, avoiding `gather_logical_kv_rows` entirely.
2. Defer until Candidate 1 parity and benchmark evidence are established.

## Validation

- `python3 -m py_compile python/sglang/srt/mem_cache/kv_page_tags.py sgl-kernel/python/sgl_kernel/kvcacheio.py sgl-kernel/tests/test_kv_checksum.py`
- CUDA: `python3 -m pytest sgl-kernel/tests/test_kv_checksum.py -q`
- Existing focused tests: `python3 -m pytest -q test/registered/unit/mem_cache/test_kv_page_tags.py test/registered/unit/disaggregation/test_kv_transfer_checksums.py test/registered/unit/disaggregation/test_kv_protection_gating.py test/registered/unit/disaggregation/test_kv_protection_abort.py`
- `git diff --check`

## Evaluation

- Run a microbenchmark comparing `hash_rows_with_positions` fallback vs `sgl_kernel.kvcacheio.kv_checksum` for `(1024,4096)`, `(4096,4096)`, `(4096,16384)`, and `(8192,16384)` byte rows.

## Promotion

- Promote Candidate 1 only if CUDA parity passes, fallback works, and benchmark evidence shows lower checksum latency or an acceptable foundation for Candidate 2.
