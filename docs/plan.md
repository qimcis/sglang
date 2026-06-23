# Executable Plan

<!--
Kernel Design Agents (KDA) executable plan. Multiple task sections may live
here. Do NOT erase unrelated sections.
-->

## KDA Task: Fused CUDA KV transfer checksum for PD disaggregation

Goal: move the `always_full`/sampled KV transfer checksum off the per-lane
Python/Torch path onto a single fused CUDA op, with bit-for-bit parity and a
safe Torch fallback.

### Step 0 — Workspace (done by orchestrator)

- Fresh worktree from `kv-page-token-tags-pd-disagg` at `e48264e6f`.
- Do not modify `/Users/chimcisaac/kernel-design-agents` (reference only).
- Do not touch unrelated untracked files in the main workspace.

### Step 1 — CUDA op (`candidate-001`)

File: `sgl-kernel/csrc/kvcacheio/checksum.cu`.

`at::Tensor kv_checksum(rows, row_indices, positions?, num_lanes)`:
- `rows` contiguous, viewed as raw bytes; `row_bytes = rows.nbytes / rows.size(0)`.
- One CUDA thread per selected row `i = row_indices[t]`:
  - `acc = CKSUM_SEED (u64)`.
  - if `positions` present: `acc = splitmix64(acc ^ (u64)positions[t])`.
  - for `j in [0, num_lanes)`: load little-endian int64 lane `j` from
    `rows + i*row_bytes` (int64 fast path when 8-byte aligned & fully in-row;
    byte assembly with zero-pad otherwise), `acc = splitmix64(acc ^ lane)`.
  - `atomicXor((unsigned long long*)out, acc)`.
- Returns a 1-element int64 tensor holding `combined = XOR_t acc_t`.
- The two scalar finishing mixes `mix(CKSUM_SEED, combined)` and `mix(., M)` are
  applied in the Python wrapper for exact parity and a single host value.

### Step 2 — Build wiring

- Declare in `sgl-kernel/include/sgl_kernel_ops.h` (csrc/kvcacheio section).
- Register schema + CUDA impl in `sgl-kernel/csrc/common_extension.cc`.
- Add `csrc/kvcacheio/checksum.cu` to `sgl-kernel/CMakeLists.txt`.

### Step 3 — Python wrapper

File: `sgl-kernel/python/sgl_kernel/kvcacheio.py`, function `kv_checksum(...)`:
- `.contiguous()` on `rows`, `row_indices`, `positions`.
- compute `row_bytes`, `total_lanes`, effective lanes.
- call op, read the single int64 back, apply the two scalar mixes, return uint64.
- Export from `sgl-kernel/python/sgl_kernel/__init__.py`.

### Step 4 — Runtime integration with fallback

File: `python/sglang/srt/mem_cache/kv_page_tags.py`:
- `_try_cuda_checksum(rows, row_indices, positions, num_lanes) -> Optional[int]`:
  returns `None` unless `rows.is_cuda`, op import succeeds, `rows.dim() <= 2`,
  and the call succeeds; any exception -> `None` (fallback).
- `hash_kv_rows`: try CUDA (kernel does the gather, avoiding `index_select`),
  else Torch path via `_hash_rows_with_positions_torch`.
- `hash_rows_with_positions`: try CUDA with identity indices, else Torch.

### Step 5 — Tests

File: `sgl-kernel/tests/test_kv_checksum.py` (CUDA-gated parity) +
extend CPU coverage indirectly through existing focused suites:
- direct op parity vs Torch (full / partial lanes / empty / non-contiguous
  indices / no-position / 1D & 2D rows / signed round-trip).
- integration parity through `hash_kv_rows` / `hash_rows_with_positions`.
- fallback: CPU tensor, missing op, runtime failure -> Torch result.

### Step 6 — Validation

- `python3 -m py_compile python/sglang/srt/mem_cache/kv_page_tags.py sgl-kernel/python/sgl_kernel/kvcacheio.py sgl-kernel/tests/test_kv_checksum.py`
- `python3 -m pytest -q test/registered/unit/mem_cache/test_kv_page_tags.py test/registered/unit/disaggregation/test_kv_transfer_checksums.py test/registered/unit/disaggregation/test_kv_protection_gating.py test/registered/unit/disaggregation/test_kv_protection_abort.py`
- `git diff --check`
- CUDA-only (H100 pod): build sgl-kernel, then
  `python3 -m pytest sgl-kernel/tests/test_kv_checksum.py -q`.

### Step 7 — Evaluation (H100 pod)

```
# from repo root on an H100 pod with sgl-kernel built/installed
pip install -e sgl-kernel    # or the project's build command
python3 sgl-kernel/benchmark/bench_kv_checksum.py
```
Record Torch vs CUDA latency for `(1024,4096)`, `(4096,4096)`, `(4096,16384)`,
`(8192,16384)` byte rows into `benchmark.csv`.

### Step 8 — Evidence + promotion

- Append candidate rows to `candidates.jsonl`.
- Write benchmark rows to `benchmark.csv`.
- Promote only if parity passes and benchmark shows improvement; otherwise leave
  unpromoted with the explicit blocker (no local CUDA).
