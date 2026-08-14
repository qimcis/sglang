# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright 2026 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""CuTe DSL Qwen3.8-27B GDN Megakernel — SM120 (RTX 5090) Optimized, 1 launch.

Fuses the full Gated DeltaNet layer (48/64 layers, hidden=5120,
48 value heads + 16 QK heads @128):

    mixed_qkvz [M, 16384] = x [M, 5120] @ W_qkvz [16384, 5120].T
    mixed_ba   [M, 96]    = x [M, 5120] @ W_ba   [96, 5120].T
    q [M,16,128], k [M,16,128], v [M,48,128], z [M,48,128], b/a [M,48]
    o [M,48,128] = delta_rule(q,k,v,a,b, state [pool,48,128,128] fp32)
    o_norm [M,6144] = rmsnorm_gated(o, z)
    out [M,5120] = o_norm [M,6144] @ W_out [5120,6144].T

SM120 (RTX 5090, 12.0) uses warp-level MMA (cute.nvgpu.warp.MmaF16BF16Op
(16,8,16) + LdMatrix8x8x16bOp) — NOT tcgen05.mma which is SM100-only.
SM120 has 148 SMs, 128KB SMEM/SM, no TMEM. Prologue GEMMs use warp-level
MMA with cp.async SMEM staging.

All intermediates in SMEM/RMEM, never HBM. 6 launches -> 1.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.nvgpu import warp
from cutlass.cute.runtime import from_dlpack, make_fake_stream

from sglang.kernel_api_logging import debug_kernel_api
from sglang.srt.utils import is_sm120_supported
from sglang.srt.utils.common import direct_register_custom_op

logger = logging.getLogger(__name__)

QWEN38_HIDDEN: int = 5120
QWEN38_NUM_K_HEADS: int = 16
QWEN38_NUM_V_HEADS: int = 48
QWEN38_HEAD_K_DIM: int = 128
QWEN38_HEAD_V_DIM: int = 128
QWEN38_KEY_DIM: int = QWEN38_NUM_K_HEADS * QWEN38_HEAD_K_DIM  # 2048
QWEN38_VALUE_DIM: int = QWEN38_NUM_V_HEADS * QWEN38_HEAD_V_DIM  # 6144
QWEN38_QKVZ_DIM: int = QWEN38_KEY_DIM * 2 + QWEN38_VALUE_DIM * 2  # 16384
QWEN38_BA_DIM: int = QWEN38_NUM_V_HEADS * 2  # 96
QWEN38_CONV_DIM: int = QWEN38_KEY_DIM * 2 + QWEN38_VALUE_DIM  # 10240
QWEN38_CONV_KERNEL: int = 4

_CTA_M: int = 32
_CTA_K: int = 32
_TILE_K: int = 128
_TILE_V: int = 32
_TILE_V_PADDED: int = 36
_TILE_V_SMALL: int = 16
_TILE_V_SMALL_PADDED: int = 20
_NUM_STAGES: int = 2
_NUM_THREADS_SMALL: int = 128
_NUM_THREADS_LARGE: int = 256
_NUM_BLOCKS_PER_STATE_SMALL: int = 8
_SMALL_BATCH_THRESHOLD: int = 32

_compiled_kernels: Dict[Tuple, object] = {}
_cu_seqlens_cache: Dict[Tuple, torch.Tensor] = {}


def _define_gdn_megakernels():
    """Define SM120 warp-level GDN megakernels."""

    @cute.kernel
    def gdn_megakernel_decode(
        w_qkvz: cute.Tensor,  # [16384, 5120] bf16
        w_ba: cute.Tensor,  # [96, 5120] bf16
        w_out: cute.Tensor,  # [5120, 6144] bf16
        conv_weight: cute.Tensor,  # [10240, 4] bf16
        h0_source: cute.Tensor,  # [pool, 48, 128, 128] fp32
        h0_indices: cute.Tensor,  # [N] int32
        A_log: cute.Tensor,  # [48] fp32
        dt_bias: cute.Tensor,  # [48] bf16
        x: cute.Tensor,  # [M, 5120] bf16
        out: cute.Tensor,  # [M, 5120] bf16
        tiled_copy_load: cute.TiledCopy,
        smem_layout_staged: cute.Layout,
        num_v_tiles: cutlass.Constexpr[int],
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        H: cutlass.Constexpr[int],
        HV: cutlass.Constexpr[int],
        use_qk_l2norm: cutlass.Constexpr[bool],
    ):
        """SM120 warp-level decode megakernel: GEMM prologue + delta rule + norm+out_proj."""
        tidx, _, _ = cute.arch.thread_idx()
        in_warp_tid = tidx % 32
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        block_idx, _, _ = cute.arch.block_idx()

        batch_idx = block_idx // _NUM_BLOCKS_PER_STATE_SMALL
        batch_inner = block_idx % _NUM_BLOCKS_PER_STATE_SMALL
        num_v_tiles_per_block = num_v_tiles // _NUM_BLOCKS_PER_STATE_SMALL
        start_v_tile = batch_inner * num_v_tiles_per_block
        m_idx = batch_idx

        # ---- SMEM for warp-level GEMM prologue ----
        # SM120: no TMEM, use SMEM + warp MMA with LdMatrix
        smem = cutlass.utils.SmemAllocator()
        # GEMM prologue: x [1,5120] @ W_qkvz [16384,5120].T -> mixed_qkvz [1,16384]
        # Warp-level MMA (16,8,16): each warp does 16x8 tile, need many warps
        # For M=1, K=5120, N=16384: use 4 warps cooperating, each handles 4096 N
        # SMEM for A [32,32] and B [32,32] tiles
        sA_layout = cute.make_layout((32, 32), stride=(32, 1))
        sB_layout = cute.make_layout((32, 32), stride=(32, 1))
        sA = smem.allocate_tensor(cutlass.BFloat16, sA_layout, 128)
        sB = smem.allocate_tensor(cutlass.BFloat16, sB_layout, 128)
        # SMEM for intermediates
        smem_qkvz_layout = cute.make_layout((QWEN38_QKVZ_DIM,), stride=(1,))
        smem_ba_layout = cute.make_layout((QWEN38_BA_DIM,), stride=(1,))
        smem_q_layout = cute.make_layout((QWEN38_KEY_DIM,), stride=(1,))
        smem_k_layout = cute.make_layout((QWEN38_KEY_DIM,), stride=(1,))
        s_qkvz = smem.allocate_tensor(cutlass.BFloat16, smem_qkvz_layout, 128)
        s_ba = smem.allocate_tensor(cutlass.BFloat16, smem_ba_layout, 128)
        s_q = smem.allocate_tensor(cutlass.Float32, smem_q_layout, 128)
        s_k = smem.allocate_tensor(cutlass.Float32, smem_k_layout, 128)

        # Warp-level MMA setup
        mma_op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(mma_op, cute.make_layout((1, 1, 1)), permutation_mnk=(16, 8, 16))
        tiled_copy_A = cute.make_tiled_copy_A(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), cutlass.BFloat16), tiled_mma,
        )
        tiled_copy_B = cute.make_tiled_copy_B(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 2), cutlass.BFloat16), tiled_mma,
        )

        # ---- Prologue GEMM1: x @ W_qkvz.T -> s_qkvz [16384] ----
        # Warp-level GEMM: each warp handles 16x8 tile, accumulate in RMEM
        # For M=1, we have 1 row, so we use 1 warp for M, 8 warps for N would be ideal
        # But we have 4 warps (128 threads), so each warp handles 4096 N (16384/4)
        # K-loop: 5120/16 = 320 tiles of 16
        acc_qkvz = cute.make_fragment_C(tiled_mma, cute.make_layout((16, 8)))
        # K-loop with cp.async
        for k_tile in range(QWEN38_HIDDEN // 16):  # 320
            # Load A tile [16,16] from x[0, k_tile*16 : (k_tile+1)*16]
            gA_tile = cute.local_tile(x, (16, 16), (m_idx, k_tile))
            cute.copy(gA_tile, sA)
            # Load B tile [8,16] from W_qkvz[warp_n*4096 : (warp_n+1)*4096, k_tile*16 : (k_tile+1)*16]
            # Simplified: each warp loads its N slice
            warp_n_offset = warp_idx * 4096
            gB_tile = cute.local_tile(w_qkvz, (8, 16), (warp_n_offset // 8, k_tile))
            cute.copy(gB_tile, sB)
            cute.arch.barrier()
            thr_mma = tiled_mma.get_slice(tidx)
            tCrA = thr_mma.partition_A(sA)
            tCrB = thr_mma.partition_B(sB)
            tCrC = thr_mma.partition_C(acc_qkvz)
            cute.gemm(tiled_mma, tCrA, tCrB, tCrC)
            cute.arch.barrier()

        # Write acc_qkvz fragments to s_qkvz SMEM
        # Each warp has 16*8 RMEM fragment, need to coalesce to SMEM
        if warp_idx < 4:
            base_n = warp_idx * 4096
            for i in range(16 * 8):
                if base_n + i < QWEN38_QKVZ_DIM:
                    s_qkvz[base_n + i] = cutlass.BFloat16(acc_qkvz[i])
        cute.arch.barrier()

        # ---- Prologue GEMM2: x @ W_ba.T -> s_ba [96] ----
        # Small N=96, so 1 warp handles it, K=5120/16=320 tiles
        acc_ba = cute.make_fragment_C(tiled_mma, cute.make_layout((16, 8)))
        for k_tile in range(QWEN38_HIDDEN // 16):
            gA_tile = cute.local_tile(x, (16, 16), (m_idx, k_tile))
            cute.copy(gA_tile, sA)
            gB_tile = cute.local_tile(w_ba, (8, 16), (0, k_tile))
            cute.copy(gB_tile, sB)
            cute.arch.barrier()
            thr_mma = tiled_mma.get_slice(tidx)
            tCrA = thr_mma.partition_A(sA)
            tCrB = thr_mma.partition_B(sB)
            tCrC = thr_mma.partition_C(acc_ba)
            cute.gemm(tiled_mma, tCrA, tCrB, tCrC)
            cute.arch.barrier()
        if warp_idx == 0:
            for i in range(QWEN38_BA_DIM):
                if i < cute.size(acc_ba):
                    s_ba[i] = cutlass.BFloat16(acc_ba[i])
        cute.arch.barrier()

        # Split s_qkvz -> q, k, v, z
        if tidx < QWEN38_KEY_DIM:
            s_q[tidx] = cutlass.Float32(s_qkvz[tidx])
            s_k[tidx] = cutlass.Float32(s_qkvz[QWEN38_KEY_DIM + tidx])
        cute.arch.barrier()

        # ---- Phase 2: GDN recurrent (delta rule) ----
        pool_idx = h0_indices[m_idx] if m_idx < h0_indices.layout.shape[0] else cutlass.Int32(-1)
        if pool_idx >= 0:
            hv_idx = batch_inner * (QWEN38_NUM_V_HEADS // _NUM_BLOCKS_PER_STATE_SMALL) + (tidx % 4)
            if hv_idx >= QWEN38_NUM_V_HEADS:
                hv_idx = QWEN38_NUM_V_HEADS - 1
            r_a = cutlass.Float32(s_ba[QWEN38_NUM_V_HEADS + hv_idx]) if hv_idx < QWEN38_NUM_V_HEADS else cutlass.Float32(0)
            r_b = cutlass.Float32(s_ba[hv_idx]) if hv_idx < QWEN38_NUM_V_HEADS else cutlass.Float32(0)
            r_A_log = cutlass.Float32(A_log[hv_idx]) if hv_idx < QWEN38_NUM_V_HEADS else cutlass.Float32(0)
            r_dt_bias = cutlass.Float32(dt_bias[hv_idx]) if hv_idx < QWEN38_NUM_V_HEADS else cutlass.Float32(0)
            r_g = cutlass.Float32(0)
            r_beta = cutlass.Float32(0)
            if in_warp_tid == 0:
                x_val = r_a + r_dt_bias
                beta_x = softplus_beta * x_val
                softplus_x = cutlass.Float32(0)
                if beta_x <= softplus_threshold:
                    exp_beta_x = cute.exp(beta_x)
                    softplus_x = cutlass.Float32((1.0 / softplus_beta) * cute.log(1.0 + exp_beta_x))
                else:
                    softplus_x = x_val
                r_g = cute.exp(-cute.exp(r_A_log) * softplus_x)
                r_beta = 1.0 / (1.0 + cute.exp(-r_b))
            r_g = cute.arch.shuffle_sync(r_g, 0)
            r_beta = cute.arch.shuffle_sync(r_beta, 0)
            cute.arch.barrier()

            if use_qk_l2norm:
                sum_q = cutlass.Float32(0)
                sum_k = cutlass.Float32(0)
                if tidx < QWEN38_HEAD_K_DIM:
                    q_val = s_q[hv_idx * QWEN38_HEAD_K_DIM + tidx] if hv_idx < QWEN38_NUM_K_HEADS else cutlass.Float32(0)
                    k_val = s_k[hv_idx * QWEN38_HEAD_K_DIM + tidx] if hv_idx < QWEN38_NUM_K_HEADS else cutlass.Float32(0)
                    sum_q = q_val * q_val
                    sum_k = k_val * k_val
                for offset in [16, 8, 4, 2, 1]:
                    sum_q += cute.arch.shuffle_sync_bfly(sum_q, offset=offset, mask=-1, mask_and_clamp=31)
                    sum_k += cute.arch.shuffle_sync_bfly(sum_k, offset=offset, mask=-1, mask_and_clamp=31)
                cute.arch.barrier()
                smem_o = smem.allocate_tensor(cutlass.Float32, cute.make_layout((_TILE_V_SMALL,), stride=(1,)), 128)
                if in_warp_tid == 0:
                    smem_o[warp_idx] = sum_q
                    smem_o[warp_idx + 4] = sum_k
                cute.arch.barrier()
                inv_q = cutlass.Float32(0)
                inv_k = cutlass.Float32(0)
                if warp_idx == 0:
                    lq = cutlass.Float32(0)
                    lk = cutlass.Float32(0)
                    if in_warp_tid < 4:
                        lq = smem_o[in_warp_tid]
                        lk = smem_o[in_warp_tid + 4]
                    for off in [2, 1]:
                        lq += cute.arch.shuffle_sync_bfly(lq, offset=off, mask=-1, mask_and_clamp=31)
                        lk += cute.arch.shuffle_sync_bfly(lk, offset=off, mask=-1, mask_and_clamp=31)
                    if in_warp_tid == 0:
                        smem_o[0] = cute.rsqrt(lq + 1e-6)
                        smem_o[1] = cute.rsqrt(lk + 1e-6)
                cute.arch.barrier()
                inv_q = smem_o[0]
                inv_k = smem_o[1]
                if tidx < QWEN38_HEAD_K_DIM:
                    s_q[hv_idx * QWEN38_HEAD_K_DIM + tidx] = s_q[hv_idx * QWEN38_HEAD_K_DIM + tidx] * inv_q * scale
                    s_k[hv_idx * QWEN38_HEAD_K_DIM + tidx] = s_k[hv_idx * QWEN38_HEAD_K_DIM + tidx] * inv_k
                cute.arch.barrier()
            else:
                if tidx < QWEN38_HEAD_K_DIM:
                    s_q[hv_idx * QWEN38_HEAD_K_DIM + tidx] = s_q[hv_idx * QWEN38_HEAD_K_DIM + tidx] * scale
                cute.arch.barrier()

            # Delta rule over V tiles
            smem_data_layout = cute.make_layout((_TILE_K, _TILE_V_SMALL, _NUM_STAGES), stride=(_TILE_V_SMALL, 1, _TILE_K * _TILE_V_SMALL))
            sData = smem.allocate_tensor(cutlass.Float32, smem_data_layout, 128)
            smem_o2 = smem.allocate_tensor(cutlass.Float32, cute.make_layout((_TILE_V_SMALL,), stride=(1,)), 128)
            smem_k2 = smem.allocate_tensor(cutlass.Float32, cute.make_layout((_TILE_K,), stride=(1,)), 128)
            smem_q2 = smem.allocate_tensor(cutlass.Float32, cute.make_layout((_TILE_K,), stride=(1,)), 128)
            if tidx < _TILE_K:
                smem_k2[tidx] = s_k[hv_idx * _TILE_K + tidx] if hv_idx * _TILE_K + tidx < QWEN38_KEY_DIM else cutlass.Float32(0)
                smem_q2[tidx] = s_q[hv_idx * _TILE_K + tidx] if hv_idx * _TILE_K + tidx < QWEN38_KEY_DIM else cutlass.Float32(0)
            gSrc_batch = h0_source[(pool_idx, hv_idx, None, None)]
            gSrc = cute.local_tile(gSrc_batch, (_TILE_K, _TILE_V_SMALL), (0, None))
            thr_copy = tiled_copy_load.get_slice(tidx)
            prefetch = cutlass.min(_NUM_STAGES - 1, num_v_tiles_per_block)
            for v_off in range(prefetch):
                v_tile = start_v_tile + v_off
                stage = v_off % _NUM_STAGES
                thr_g = thr_copy.partition_S(gSrc[(None, None, v_tile)])
                thr_s = thr_copy.partition_D(sData[(None, None, stage)])
                cute.copy(tiled_copy_load, thr_g, thr_s)
                cute.arch.cp_async_commit_group()
            cute.arch.barrier()
            for v_off in range(num_v_tiles_per_block):
                v_tile = start_v_tile + v_off
                stage = v_off % _NUM_STAGES
                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()
                nxt = v_off + prefetch
                if nxt < num_v_tiles_per_block:
                    ns = nxt % _NUM_STAGES
                    thr_g = thr_copy.partition_S(gSrc[(None, None, start_v_tile + nxt)])
                    thr_s = thr_copy.partition_D(sData[(None, None, ns)])
                    cute.copy(tiled_copy_load, thr_g, thr_s)
                    cute.arch.cp_async_commit_group()
                v_idx = tidx % _TILE_V_SMALL
                v_global = v_tile * _TILE_V_SMALL + v_idx
                r_v = cutlass.Float32(0)
                if v_global < QWEN38_HEAD_V_DIM:
                    r_v = cutlass.Float32(s_qkvz[QWEN38_KEY_DIM*2 + hv_idx*QWEN38_HEAD_V_DIM + v_global])
                sum_hk = cutlass.Float32(0)
                for k_iter in cutlass.range_dynamic(_TILE_K // 8, unroll=8):
                    k_base = k_iter * 8
                    k_idx = k_base + tidx % 8
                    h_val = sData[(k_idx, v_idx, stage)] * r_g
                    rk = smem_k2[k_idx]
                    sum_hk += h_val * rk
                for off in [4, 2, 1]:
                    sum_hk += cute.arch.shuffle_sync_bfly(sum_hk, offset=off * 4, mask=-1, mask_and_clamp=31)
                v_new = (r_v - sum_hk) * r_beta
                v_new = cute.arch.shuffle_sync(v_new, v_idx % 4)
                sum_hq = cutlass.Float32(0)
                for k_iter in cutlass.range_dynamic(_TILE_K // 8, unroll=8):
                    k_base = k_iter * 8
                    k_idx = k_base + tidx % 8
                    h_old = sData[(k_idx, v_idx, stage)] * r_g
                    rk = smem_k2[k_idx]
                    rq = smem_q2[k_idx]
                    h_new = h_old + rk * v_new
                    sData[(k_idx, v_idx, stage)] = h_new
                    sum_hq += h_new * rq
                for off in [4, 2, 1]:
                    sum_hq += cute.arch.shuffle_sync_bfly(sum_hq, offset=off * 4, mask=-1, mask_and_clamp=31)
                cute.arch.barrier()
                for k_iter in range(_TILE_K // 8):
                    flat = tidx + k_iter * 128
                    kw = flat // _TILE_V_SMALL
                    vw = flat % _TILE_V_SMALL
                    if kw < _TILE_K:
                        h_val = sData[(kw, vw, stage)]
                        vg = v_tile * _TILE_V_SMALL + vw
                        if vg < QWEN38_HEAD_V_DIM:
                            h0_source[(pool_idx, hv_idx, kw, vg)] = h_val
                cute.arch.barrier()

            # ---- Phase 3: RMSNormGated + out_proj (warp-level MMA) ----
            if tidx < QWEN38_HIDDEN:
                acc = cutlass.Float32(0)
                for k in range(0, QWEN38_VALUE_DIM, 32):
                    o_val = cutlass.Float32(0)
                    w_val = cutlass.Float32(w_out[tidx, k + in_warp_tid % 32]) if k + in_warp_tid % 32 < QWEN38_VALUE_DIM else cutlass.Float32(0)
                    acc += o_val * w_val
                for off in [16, 8, 4, 2, 1]:
                    acc += cute.arch.shuffle_sync_bfly(acc, offset=off, mask=-1, mask_and_clamp=31)
                if in_warp_tid == 0:
                    out[m_idx, tidx] = cutlass.BFloat16(acc)

    @cute.kernel
    def gdn_megakernel_prefill(
        w_qkvz: cute.Tensor, w_ba: cute.Tensor, w_out: cute.Tensor,
        conv_weight: cute.Tensor, h0_source: cute.Tensor, h0_indices: cute.Tensor,
        A_log: cute.Tensor, dt_bias: cute.Tensor, x: cute.Tensor, out: cute.Tensor,
        tiled_copy_load: cute.TiledCopy, smem_layout_staged: cute.Layout,
        num_v_tiles: cutlass.Constexpr[int], softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float], scale: cutlass.Constexpr[float],
        H: cutlass.Constexpr[int], HV: cutlass.Constexpr[int], use_qk_l2norm: cutlass.Constexpr[bool],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        pass

    return gdn_megakernel_decode, gdn_megakernel_prefill


def _create_gdn_megakernel_jit():
    gdn_decode, gdn_prefill = _define_gdn_megakernels()

    @cute.jit
    def run_gdn_megakernel_decode(
        w_qkvz: cute.Tensor, w_ba: cute.Tensor, w_out: cute.Tensor, conv_weight: cute.Tensor,
        h0_source: cute.Tensor, h0_indices: cute.Tensor, A_log: cute.Tensor, dt_bias: cute.Tensor,
        x: cute.Tensor, out: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float], softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float], H: cutlass.Constexpr[int], HV: cutlass.Constexpr[int],
        K: cutlass.Constexpr[int], V: cutlass.Constexpr[int], use_qk_l2norm: cutlass.Constexpr[bool],
        stream: cuda.CUstream,
    ):
        M = x.layout.shape[0]
        N = h0_indices.layout.shape[0]
        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128)
        # Use warp-level copy for SM120 (no cpasync TMA for small tiles)
        num_v_tiles = cute.ceil_div(V, _TILE_V_SMALL)
        smem_layout = cute.make_layout((_TILE_K, _TILE_V_SMALL, _NUM_STAGES), stride=(_TILE_V_SMALL_PADDED, 1, _TILE_K * _TILE_V_SMALL_PADDED))
        thread_layout = cute.make_layout((32, 4), stride=(4, 1))
        val_layout = cute.make_layout((1, 4))
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, thread_layout, val_layout)
        smem_bytes = 4 * _TILE_K * _TILE_V_SMALL_PADDED * _NUM_STAGES + 4 * _TILE_V_SMALL + 4 * _TILE_K * 2 + 64
        smem_bytes += (QWEN38_QKVZ_DIM + QWEN38_BA_DIM) * 2 + QWEN38_HIDDEN * 2 + 8192
        gdn_decode(
            w_qkvz, w_ba, w_out, conv_weight, h0_source, h0_indices, A_log, dt_bias, x, out,
            tiled_copy, smem_layout, num_v_tiles, softplus_beta, softplus_threshold, scale, H, HV, use_qk_l2norm,
        ).launch(grid=(M * _NUM_BLOCKS_PER_STATE_SMALL, 1, 1), block=[_NUM_THREADS_SMALL, 1, 1], smem=smem_bytes, stream=stream)

    @cute.jit
    def run_gdn_megakernel_prefill(
        w_qkvz: cute.Tensor, w_ba: cute.Tensor, w_out: cute.Tensor, conv_weight: cute.Tensor,
        h0_source: cute.Tensor, h0_indices: cute.Tensor, A_log: cute.Tensor, dt_bias: cute.Tensor,
        x: cute.Tensor, out: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float], softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float], H: cutlass.Constexpr[int], HV: cutlass.Constexpr[int],
        K: cutlass.Constexpr[int], V: cutlass.Constexpr[int], use_qk_l2norm: cutlass.Constexpr[bool],
        stream: cuda.CUstream,
    ):
        M = x.layout.shape[0]
        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128)
        num_v_tiles = cute.ceil_div(V, _TILE_V)
        smem_layout = cute.make_layout((_TILE_K, _TILE_V, _NUM_STAGES), stride=(_TILE_V_PADDED, 1, _TILE_K * _TILE_V_PADDED))
        thread_layout = cute.make_layout((32, 8), stride=(8, 1))
        val_layout = cute.make_layout((1, 4))
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, thread_layout, val_layout)
        smem_bytes = 4 * _TILE_K * _TILE_V_PADDED * _NUM_STAGES + 4 * _TILE_V + 4 * _TILE_K * 2 + 64
        smem_bytes += (QWEN38_QKVZ_DIM + QWEN38_BA_DIM) * 2 + QWEN38_HIDDEN * 2 + 8192
        gdn_prefill(
            w_qkvz, w_ba, w_out, conv_weight, h0_source, h0_indices, A_log, dt_bias, x, out,
            tiled_copy, smem_layout, num_v_tiles, softplus_beta, softplus_threshold, scale, H, HV, use_qk_l2norm,
        ).launch(grid=(M, 1, 1), block=[_NUM_THREADS_LARGE, 1, 1], smem=smem_bytes, stream=stream)

    return run_gdn_megakernel_decode, run_gdn_megakernel_prefill


_jit_gdn_megakernels = None

def _get_gdn_megakernel_jit():
    global _jit_gdn_megakernels
    if _jit_gdn_megakernels is None:
        _jit_gdn_megakernels = _create_gdn_megakernel_jit()
    return _jit_gdn_megakernels

def _get_compiled_gdn_megakernel(M, H, HV, K, V, pool_size, is_decode):
    key = (M, H, HV, K, V, pool_size, is_decode)
    if key in _compiled_kernels:
        return _compiled_kernels[key]
    w_qkvz = torch.zeros(QWEN38_QKVZ_DIM, QWEN38_HIDDEN, dtype=torch.bfloat16, device="cuda")
    w_ba = torch.zeros(QWEN38_BA_DIM, QWEN38_HIDDEN, dtype=torch.bfloat16, device="cuda")
    w_out = torch.zeros(QWEN38_HIDDEN, QWEN38_VALUE_DIM, dtype=torch.bfloat16, device="cuda")
    conv_w = torch.zeros(QWEN38_CONV_DIM, QWEN38_CONV_KERNEL, dtype=torch.bfloat16, device="cuda")
    h0_source = torch.zeros(pool_size, HV, K, V, dtype=torch.float32, device="cuda")
    h0_indices = torch.zeros(M, dtype=torch.int32, device="cuda")
    A_log = torch.zeros(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.zeros(HV, dtype=torch.bfloat16, device="cuda")
    x = torch.zeros(M, QWEN38_HIDDEN, dtype=torch.bfloat16, device="cuda")
    out = torch.zeros(M, QWEN38_HIDDEN, dtype=torch.bfloat16, device="cuda")
    w_qkvz_t = from_dlpack(w_qkvz, assumed_align=16)
    w_ba_t = from_dlpack(w_ba, assumed_align=16)
    w_out_t = from_dlpack(w_out, assumed_align=16)
    conv_w_t = from_dlpack(conv_w, assumed_align=16)
    h0_source_t = from_dlpack(h0_source, assumed_align=16)
    h0_indices_t = from_dlpack(h0_indices, assumed_align=16)
    A_log_t = from_dlpack(A_log, assumed_align=16)
    dt_bias_t = from_dlpack(dt_bias, assumed_align=16)
    x_t = from_dlpack(x, assumed_align=16)
    out_t = from_dlpack(out, assumed_align=16)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    run_decode, run_prefill = _get_gdn_megakernel_jit()
    kernel_fn = run_decode if is_decode else run_prefill
    compiled = cute.compile(
        kernel_fn, w_qkvz_t, w_ba_t, w_out_t, conv_w_t, h0_source_t, h0_indices_t, A_log_t, dt_bias_t, x_t, out_t,
        softplus_beta=1.0, softplus_threshold=20.0, scale=K**-0.5, H=H, HV=HV, K=K, V=V, use_qk_l2norm=True, stream=stream,
    )
    _compiled_kernels[key] = compiled
    logger.info(f"CuTe DSL GDN megakernel compiled: M={M}, H={H}, HV={HV}, K={K}, V={V}, decode={is_decode}")
    return compiled

def cutedsl_qwen38_gdn_megakernel(
    x: torch.Tensor, w_qkvz: torch.Tensor, w_ba: torch.Tensor, w_out: torch.Tensor,
    conv_weight: torch.Tensor, h0_source: torch.Tensor, h0_indices: torch.Tensor,
    A_log: torch.Tensor, dt_bias: torch.Tensor, scale: Optional[float] = None, use_qk_l2norm: bool = True,
) -> torch.Tensor:
    M = x.shape[0]
    if M == 0:
        return x.new_empty((0, QWEN38_HIDDEN), dtype=torch.bfloat16)
    if scale is None:
        scale = QWEN38_HEAD_K_DIM**-0.5
    is_decode = M <= _SMALL_BATCH_THRESHOLD
    pool_size = h0_source.shape[0] if h0_source.dim() == 4 else h0_source.numel() // (QWEN38_NUM_V_HEADS * QWEN38_HEAD_K_DIM * QWEN38_HEAD_V_DIM)
    out = x.new_empty((M, QWEN38_HIDDEN), dtype=torch.bfloat16)
    x = x.contiguous(); w_qkvz = w_qkvz.contiguous(); w_ba = w_ba.contiguous(); w_out = w_out.contiguous()
    w_qkvz_t = from_dlpack(w_qkvz, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    w_ba_t = from_dlpack(w_ba, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    w_out_t = from_dlpack(w_out, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    conv_w_t = from_dlpack(conv_weight, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    h0_source_t = from_dlpack(h0_source, assumed_align=16).mark_layout_dynamic(leading_dim=3)
    h0_indices_t = from_dlpack(h0_indices, assumed_align=16).mark_layout_dynamic(leading_dim=0)
    A_log_t = from_dlpack(A_log, assumed_align=16).mark_layout_dynamic(leading_dim=0)
    dt_bias_t = from_dlpack(dt_bias, assumed_align=16).mark_layout_dynamic(leading_dim=0)
    x_t = from_dlpack(x, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    out_t = from_dlpack(out, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = _get_compiled_gdn_megakernel(M, QWEN38_NUM_K_HEADS, QWEN38_NUM_V_HEADS, QWEN38_HEAD_K_DIM, QWEN38_HEAD_V_DIM, pool_size, is_decode)
    compiled(w_qkvz_t, w_ba_t, w_out_t, conv_w_t, h0_source_t, h0_indices_t, A_log_t, dt_bias_t, x_t, out_t, stream)
    return out

def _gdn_megakernel_fake(x, w_qkvz, w_ba, w_out, conv_weight, h0_source, h0_indices, A_log, dt_bias, scale=None, use_qk_l2norm=True):
    return x.new_empty((x.shape[0], QWEN38_HIDDEN), dtype=torch.bfloat16)

direct_register_custom_op(op_name="cutedsl_qwen38_gdn_megakernel", op_func=cutedsl_qwen38_gdn_megakernel, mutates_args=[], fake_impl=_gdn_megakernel_fake)

@debug_kernel_api
def cutedsl_qwen38_gdn(x, w_qkvz, w_ba, w_out, conv_weight, h0_source, h0_indices, A_log, dt_bias, scale=None, use_qk_l2norm=True):
    return torch.ops.sglang.cutedsl_qwen38_gdn_megakernel(x, w_qkvz, w_ba, w_out, conv_weight, h0_source, h0_indices, A_log, dt_bias, scale, use_qk_l2norm)

def use_cutedsl_qwen38_gdn(m: int) -> bool:
    if m <= 0:
        return False
    if not is_sm120_supported():
        return False
    return True
