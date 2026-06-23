# Draft Plan

<!--
This file is a Kernel Design Agents (KDA) evidence log. Multiple task sections
may live here. Do NOT erase unrelated sections; append/refine only the section
for the task you are working on.
-->

## KDA Task: Fused CUDA KV transfer checksum for PD disaggregation

### Task Contract

- Task name: Fused CUDA KV transfer checksum for PD disaggregation.
- Objective: Replace the Python/Torch checksum path in
  `python/sglang/srt/mem_cache/kv_page_tags.py` with a CUDA/C++ fused kernel
  path when inputs are CUDA tensors, preserving bit-for-bit parity and falling
  back to the existing Torch path on CPU / unsupported inputs / missing op /
  runtime failure.
- Inputs:
  - `rows`: logical KV rows tensor `[num_tokens, row_len...]`, any integer/byte
    dtype, reinterpreted as little-endian int64 lanes.
  - `row_indices`: int64 selected logical token indices into `rows` dim 0.
  - `positions` (optional): int64 logical positions folded in for order
    sensitivity.
  - `num_lanes` (optional cap): leading int64 lanes/row to hash (partial mode);
    `-1`/`None` means all lanes.
- Output: one signed int64 checksum bit-pattern matching the current
  `hash_rows_with_positions` semantics.
- Correctness requirements / invariants:
  - Bit-for-bit parity with the current Torch checksum for contiguous logical
    rows (1D/2D).
  - Never hash node-local physical page ids.
  - Preserve logical-position mixing (order sensitivity).
  - Preserve empty-selection behavior (`_splitmix64_scalar(_CKSUM_SEED)`).
  - Preserve signed int64 bit-pattern semantics used in metadata transport.
  - Fall back to the Torch path on CPU / unsupported inputs / missing op /
    runtime failure.
- Performance target: significantly reduce full checksum latency vs the Torch
  path for representative H100 byte-row shapes `(1024,4096)`, `(4096,4096)`,
  `(4096,16384)`, `(8192,16384)` (mode `always_full`).
- Allowed implementation: CUDA/C++ custom op in `sgl-kernel` (no Triton); Python
  wrapper in `sgl-kernel/python/sgl_kernel/kvcacheio.py`; runtime wiring in
  `kv_page_tags.py` with fallback; tests under `sgl-kernel/tests/`; benchmark
  under `sgl-kernel/benchmark/`.
- Validation command:
  - `python3 -m py_compile python/sglang/srt/mem_cache/kv_page_tags.py sgl-kernel/python/sgl_kernel/kvcacheio.py sgl-kernel/tests/test_kv_checksum.py`
  - `python3 -m pytest sgl-kernel/tests/test_kv_checksum.py -q` (CUDA build required)
  - `python3 -m pytest -q test/registered/unit/mem_cache/test_kv_page_tags.py test/registered/unit/disaggregation/test_kv_transfer_checksums.py test/registered/unit/disaggregation/test_kv_protection_gating.py test/registered/unit/disaggregation/test_kv_protection_abort.py`
  - `git diff --check`
- Evaluation command: `python3 sgl-kernel/benchmark/bench_kv_checksum.py` on an
  H100 (Torch vs fused CUDA for the four shapes).
- Promotion criteria: parity test exists & passes where runnable; fallback
  preserves existing behavior; CUDA op used only for safe CUDA tensors;
  benchmark shows improvement, or the patch clearly states the remaining blocker
  and is not promoted.

### Baseline

- Current implementation: `hash_kv_rows` -> `hash_rows_with_positions`. Rows are
  reinterpreted to int64 lanes by `_as_int64_lanes`; the code loops over int64
  lanes from Python, doing one `_mix_tensor` (splitmix64 over the whole token
  axis) per lane, then a log-step XOR reduce and two scalar finishing mixes.
  For `always_full` this is `O(num_lanes)` CUDA kernel launches over `num_tokens`
  elements each, which dominated full-PD-serving latency.
- Hash definition (must be matched bit-for-bit):
  - `splitmix64(x)`: `x += 0x9E3779B97F4A7C15; z=x; z=(z^(z>>30))*0xBF58476D1CE4E5B9; z=(z^(z>>27))*0x94D049BB133111EB; z=z^(z>>31)` (all mod 2^64).
  - `mix(acc, field) = splitmix64(acc ^ field)`.
  - Per selected row `i`: `acc_i = CKSUM_SEED`; if positions, `acc_i = mix(acc_i, pos_i)`; for each lane `j`: `acc_i = mix(acc_i, lane_ij)`.
  - `combined = XOR_i acc_i`.
  - `total = mix(CKSUM_SEED, combined); total = mix(total, M)` where `M` = number of selected rows.
  - Empty selection returns `splitmix64(CKSUM_SEED)`.
  - `_CKSUM_SEED = 0x5347_4C41_4E47_4353`.
- Existing tests / validators: `test/registered/unit/disaggregation/test_kv_transfer_checksums.py`, `test/registered/unit/mem_cache/test_kv_page_tags.py`.
- Baseline validation result: CPU Torch path is the reference; CUDA build not
  available on this macOS dev box.

### Risks And Unknowns

- `row_indices` / `positions` may be non-contiguous (tests use `rows[::3]`-style
  slices); the kernel must read them contiguously -> wrapper calls `.contiguous()`.
- Lane reinterpretation must match `_as_int64_lanes`: bytes are little-endian,
  the trailing partial lane is zero-padded to 8 bytes; alignment of a row start
  is not guaranteed for arbitrary `row_bytes`.
- 64-bit `atomicXor` requires sm_35+ (H100 = sm_90, fine) but must compile for
  all configured arches.
- `>2D` rows: the Torch fallback asserts 1D/2D, so the CUDA path is only
  attempted for 1D/2D rows to guarantee identical behavior.
- No CUDA on the dev box: CUDA pytest + microbenchmark must be run on an H100
  pod; this is the only remaining evidence gap before promotion.

### Candidate Directions

1. `candidate-001` (fused-gathered-rows): one CUDA op `kv_checksum(rows,
   row_indices, positions?, num_lanes)` that gathers the selected rows itself
   (no `index_select` materialization), folds positions + lanes per row, and
   XOR-reduces via 64-bit `atomicXor`. The two scalar finishing mixes stay in
   Python for exact parity. Expected: single launch replaces `O(num_lanes)`
   launches. Risk: byte-assembly cost for unaligned/partial lanes. Validation:
   parity test vs Torch on GPU; fallback tests on CPU. Evaluation: microbench
   on the four shapes.
2. `candidate-002` (hash-from-kv-loc, deferred): hash directly from the KV cache
   buffers + `kv_loc` to also remove the `gather_logical_kv_rows`
   materialization. Higher complexity (per-layer K/V, MLA folding). Deferred
   until candidate-001 parity + speedup are confirmed on H100.

### First Implementation Steps

1. Write `sgl-kernel/csrc/kvcacheio/checksum.cu` with the splitmix64 finalizer,
   little-endian lane loader (int64 fast path + byte assembly), per-row fold,
   and 64-bit `atomicXor` reduce returning a single int64 (the `combined` XOR).
2. Declare `kv_checksum` in `include/sgl_kernel_ops.h`, register in
   `common_extension.cc`, add the source to `CMakeLists.txt`.
3. Add `kv_checksum` Python wrapper in `kvcacheio.py` (contiguity + final scalar
   mixes) and export it; wire `_try_cuda_checksum` into `kv_page_tags.py`.
4. Add parity + fallback tests and the microbenchmark.

### Evidence Required For Promotion

- Correctness evidence: GPU parity test (`test_kv_checksum.py`) green; CPU
  fallback tests for the four focused suites green; bit-for-bit equality with the
  Torch reference across full/partial/empty/non-contiguous/no-position cases.
- Performance evidence: `bench_kv_checksum.py` showing speedup for the four
  shapes on H100.
- Regression checks: existing checksum/page-tag suites unchanged on CPU.
