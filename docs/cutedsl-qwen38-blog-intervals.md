# CuTeDSL Qwen3.8-27B Megakernels — Blog Intervals

> **Model:** `Qwen3.8-27B` dense hybrid GDN — `64` layers `16×[3×GDN+1×Attn]`, `hidden=5120`, `intermediate=17408`, `48v/16k heads @128`.
> **Checkpoints:** `Qwen/Qwen3.8-27B` (BF16, H200) / `RadixArk/Qwen3.8-27B-NVFP4` (W4A4+FP8, Blackwell).
> **Lowrank stuff is separate** — not covered here.

Test each interval on **SM100** (RTX 5090/6000, GB300) with `torch.cuda.nvtx` + `nsys`. Toggle via env vars.

---

## Interval 0 — Baseline (PyTorch Eager)

**What:** No megakernels. `Qwen3_5GatedDeltaNet.forward:631` (6 launches) + `Qwen2MoeMLP.forward:210` (3 BF16 / 2 NVFP4 launches).

**PyTorch impl:**
```python
# GDN (qwen3_5.py:631) — 6 launches
qkvz, ba = self.in_proj_qkvz(x), self.in_proj_ba(x)          # 2× F.linear
q, k, v, z, b, a = split_reshape(qkvz, ba)                   # 1× triton
o = self.attn(mixed_qkv, a, b)                                # 1× fused_recurrent
o = self.norm(o, z)                                           # 1× RMSNormGated
out = self.out_proj(o)                                        # 1× F.linear

# MLP (qwen2_moe.py:210) — 3 launches BF16, 2 NVFP4
gate_up, _ = self.gate_up_proj(x)    # [M,34816] = x[5120] @ W[34816,5120].T
act = self.act_fn(gate_up)           # silu(gate)*up, gate/up split in RMEM
out, _ = self.down_proj(act)         # [M,5120] = act[17408] @ W[5120,17408].T
```

**Launches/layer:** GDN 6 + MLP 3 (BF16) / 2 (NVFP4) = **9 / 8**.
**HBM/layer:** `M*16384*2B` (qkvz) + `M*34816*2B` (gate_up) + `M*6144*2B` (o_norm).
**How to test:** `SGLANG_QWEN38_GDN_MEGAKERNEL_DISABLE=1 SGLANG_QWEN38_MLP_MEGAKERNEL_DISABLE=1 python -m sglang.launch_server --model-path Qwen/Qwen3.8-27B`

---

## Interval 1 — MLP BF16 Megakernel (3→1)

**File:** `sglang/python/sglang/kernels/ops/gemm/cutedsl_qwen38_mlp.py:Qwen38MlpBf16FusedKernel`

**What:** `x @ W_gate_up.T [5120→34816] -> silu(gate)*up -> @ W_down.T [17408→5120]` in **1 launch**. `2*I=34816` intermediate in SMEM (`32×64×2B=4KB/CTA`), never HBM. `tcgen05.mma` BF16, `TMEM→RMEM` SiLU epilog.

**Saves:** `M*34816*2B` HBM/layer + 2 launches. At `M=2048`, `~136MB` for 64 layers.
**Expected:** MLP `1.25-1.35x`, e2e `1.08-1.12x` (BF16).
**How to test:** Default on for `hidden=5120, gate_up=34816` on SM100. Disable: `SGLANG_QWEN38_MLP_MEGAKERNEL_DISABLE=1`.

```python
# Before: 3 launches
gate_up = F.linear(x, w_gate_up)          # launch 1
act = F.silu(gate_up.chunk(2, -1)[0]) * gate_up.chunk(2, -1)[1]  # launch 2
out = F.linear(act, w_down)               # launch 3

# After: 1 launch
out = cutedsl_qwen38_mlp_bf16(x, w_gate_up, w_down)  # TMEM SiLU, SMEM act
```

---

## Interval 2 — MLP NVFP4 Megakernel (2→1)

**File:** `sglang/python/sglang/kernels/ops/gemm/cutedsl_qwen38_mlp.py:Qwen38MlpNvfp4FusedKernel`

**What:** Same as Interval 1 but block-scaled `Float4E2M1FN + Float8E8M0FNU sf_vec_size=16` via `make_blockscaled_trivial_tiled_mma`. `act_fp4 [M,17408*0.5B] + act_scale [M,1088B]` in SMEM between MMAs. Replaces `Sm100BlockScaledPersistentDenseGemmKernel` 2-launch (`nvfp4_gemm_swiglu_nvfp4_quant.py:114` already fuses `GEMM+SwiGLU+quant` but spills `act`).

**Saves:** `M*17408*0.5B + scales` HBM/layer + 1 launch. At `M=2048`, `~17MB` for 64 layers.
**Expected:** MLP `1.08-1.12x`, e2e `1.04-1.06x` (NVFP4). Smaller than BF16 because `0.5B` vs `2B`.
**How to test:** `RadixArk/Qwen3.8-27B-NVFP4` on SM100. Currently falls back to 2-launch until `x_scale/w_scale/alpha` wired via `quant_config` — scaffolded, needs `MergedColumnParallelLinear` scale passthrough.

```python
# Before: 2 launches (already fused gate_up+SwiGLU+quant)
act_fp4, act_scale = nvfp4_gemm_swiglu_nvfp4_quant(x_fp4, x_scale, w_gu_fp4, w_gu_scale, alpha, out_scale)  # launch 1
out = fp4_gemm(act_fp4, act_scale, w_down_fp4, w_down_scale, alpha)                                        # launch 2

# After: 1 launch
out = cutedsl_qwen38_mlp_nvfp4(x_fp4, x_scale, w_gu_fp4, w_gu_scale, w_down_fp4, w_down_scale, alpha, out_scale)  # SMEM act
```

---

## Interval 3 — GDN Megakernel (6→1, TMA+tcgen05.mma Prologue)

**File:** `sglang/python/sglang/kernels/ops/attention/cutedsl_qwen38_gdn.py`

**What:** Full GDN layer in 1 launch: `in_proj_qkvz [16384,5120] + in_proj_ba [96,5120]` via `TMA+tcgen05.mma` (40 K-tiles of 128, `tiled_mma_qkvz`/`tiled_mma_ba`, `TMEM→RMEM→SMEM` to `s_qkvz`/`s_ba`), `split/reshape` in SMEM, `delta_rule` (state `[pool,48,128,128]` fp32, `9.6 TB/s` row-streaming), `RMSNormGated` + `out_proj [5120,6144]` GEMM. Was scaffolded `cute.copy` warp-reduce, now `TMA+tcgen05.mma`.

**Saves:** `M*16384*2B + M*6144*2B` HBM/layer + 5 launches. At `M=2048`, `~80MB` for 48 GDN layers.
**Expected:** GDN `1.15-1.25x`, e2e `1.08-1.12x` (48/64 layers). Biggest single win.
**How to test:** Default on for `hidden=5120, 48v/16k` on SM100. Disable: `SGLANG_QWEN38_GDN_MEGAKERNEL_DISABLE=1`.

```python
# Before: 6 launches
qkvz, ba = self.in_proj_qkvz(x), self.in_proj_ba(x)  # 2× F.linear
q, k, v, z, b, a = split_reshape(qkvz, ba)            # 1× triton
o = self.attn(mixed_qkv, a, b)                         # 1× fused_recurrent
o = self.norm(o, z)                                    # 1× RMSNormGated
out = self.out_proj(o)                                 # 1× F.linear

# After: 1 launch
out = cutedsl_qwen38_gdn(x, w_qkvz, w_ba, w_out, conv_w, h0_source, h0_indices, A_log, dt_bias)
```

---

## Interval 4 — Persistent GDN→MLP→GDN→MLP (4 Layers, 1 Launch)

**File:** `sglang/python/sglang/kernels/ops/cutedsl_qwen38_persistent.py:persistent_gdn_mlp_kernel`

**What:** Persistent CTA pipelines `M` tiles through `GDN1→MLP1→GDN2→MLP2` (4 layers, 2 pairs). `hidden_states [M,5120]` tile `32×32=2KB` in SMEM across layers, no HBM per boundary. Grid `(M/32, 5120/32=160)`. Each CTA does 4 layers sequentially.

**Saves:** `M*5120*2B` per boundary ×3 = `M*30KB` (M=1) / `M*30MB` (M=2048) for 2 pairs.
**Expected:** `1.05-1.08x` on top of per-layer (Intervals 1+3).
**How to test:** Standalone `cutedsl_qwen38_persistent_gdn_mlp(...)` — not yet auto-dispatched in `qwen3_5.py` loop. Wire via `Qwen3_5ForCausalLM.forward` layer loop when `M>=32` and `use_cutedsl_qwen38_persistent(M)`.

```python
# Before: 4× per-layer megakernels = 4 launches
x = cutedsl_qwen38_gdn(x, *gdn1_weights)      # launch 1
x = cutedsl_qwen38_mlp_bf16(x, *mlp1_weights) # launch 2
x = cutedsl_qwen38_gdn(x, *gdn2_weights)      # launch 3
x = cutedsl_qwen38_mlp_bf16(x, *mlp2_weights) # launch 4

# After: 1 launch
x = cutedsl_qwen38_persistent_gdn_mlp(x, *gdn1_weights, *mlp1_weights, *gdn2_weights, *mlp2_weights)
```

---

## Interval 5 — Triple-GDN (3×GDN, Shared TMA)

**File:** `sglang/python/sglang/kernels/ops/cutedsl_qwen38_persistent.py:triple_gdn_kernel`

**What:** For `16×[3×GDN+1×Attn]`, fuses the 3 GDNs in each repeat. One TMA descriptor for `W_qkvz [16384,5120]` (same geometry ×3, different base pointers), L2 reuse, `hidden_states` in SMEM. Saves 2 launches + 2 HBM per repeat ×16 = 32 launches.

**Saves:** `M*5120*2B` per boundary ×2 = `M*20KB` (M=1) / `M*20MB` (M=2048) per repeat.
**Expected:** `1.03-1.05x` on top of per-layer.
**How to test:** Standalone `cutedsl_qwen38_triple_gdn(...)` — wire in `Qwen3_5ForCausalLM` when detecting `3×GDN` pattern.

```python
# Before: 3× GDN megakernels = 3 launches
x = cutedsl_qwen38_gdn(x, *gdn1_weights)  # launch 1
x = cutedsl_qwen38_gdn(x, *gdn2_weights)  # launch 2
x = cutedsl_qwen38_gdn(x, *gdn3_weights)  # launch 3

# After: 1 launch, shared TMA
x = cutedsl_qwen38_triple_gdn(x, *gdn1_weights, *gdn2_weights, *gdn3_weights)
```

---

## Interval 6 — All Together

**What:** Intervals 1+2+3+4+5. Per-layer MLP+GDN + persistent cross-layer.

**Expected (NVFP4 Blackwell, vs eager):**
- MLP 2→1: `1.04-1.06x`
- GDN 6→1: `1.08-1.12x`
- Persistent 4-layer: `1.05-1.08x` on top
- Triple-GDN: `1.03-1.05x` on top
- **Combined: `1.20-1.30x` e2e** (BF16 `1.25-1.35x`, NVFP4 `1.20-1.30x`)

**Benchmark harness:**
```python
import torch, time
from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM

# Sweep M = [1, 32, 128, 512, 2048] (decode → prefill)
for M in [1, 32, 128, 512, 2048]:
    x = torch.randn(M, 5120, dtype=torch.bfloat16, device="cuda")
    # Warmup + nsys
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        out = model.forward(x, forward_batch)
    torch.cuda.synchronize()
    print(f"M={M}: {100/(time.perf_counter()-t0):.1f} iter/s")
```

**Env toggles for ablation:**
```bash
# Baseline
SGLANG_QWEN38_GDN_MEGAKERNEL_DISABLE=1 SGLANG_QWEN38_MLP_MEGAKERNEL_DISABLE=1

# +MLP only
SGLANG_QWEN38_GDN_MEGAKERNEL_DISABLE=1

# +GDN only
SGLANG_QWEN38_MLP_MEGAKERNEL_DISABLE=1

# All on (default SM100)
python -m sglang.launch_server --model-path RadixArk/Qwen3.8-27B-NVFP4
```

---

## Files

- `sglang/python/sglang/kernels/ops/gemm/cutedsl_qwen38_mlp.py` — MLP BF16+NVFP4 1-launch
- `sglang/python/sglang/kernels/ops/attention/cutedsl_qwen38_gdn.py` — GDN 1-launch TMA+mma
- `sglang/python/sglang/kernels/ops/cutedsl_qwen38_persistent.py` — Persistent + Triple-GDN
- `sglang/python/sglang/srt/models/qwen3_5.py:660` — GDN dispatch
- `sglang/python/sglang/srt/models/qwen2_moe.py:231` — MLP dispatch
