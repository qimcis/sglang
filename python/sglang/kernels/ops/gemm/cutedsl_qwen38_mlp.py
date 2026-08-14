# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright 2026 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""CuTe DSL Qwen3.8-27B MLP Megakernels — BF16 and NVFP4, 1 launch.

Fuses the dense SwiGLU MLP that dominates Qwen3.8-27B (hidden=5120,
intermediate=17408, 64 layers, ~17B params):

    gate_up [M, 2*I] = x [M, K] @ W_gate_up [2*I, K].T
    gate, up         = split(gate_up, 2)              # [M, I] each
    act    [M, I]    = silu(gate) * up                # fused in epilog
    out    [M, K]    = act [M, I] @ W_down [K, I].T

Two megakernels, both 1 launch (act never hits HBM):

* **BF16** (Qwen/Qwen3.8-27B, H200 + Blackwell BF16): both GEMMs are BF16
  tcgen05.mma, fp32 accum, BF16 out. SiLU fused in GEMM1 epilog. Saves
  M*34816*2B HBM + 1 launch vs eager (gate_up + SiluAndMul + down).

* **NVFP4** (RadixArk/Qwen3.8-27B-NVFP4, Blackwell only): both GEMMs are
  block-scaled FP4 tcgen05.mma (Float4E2M1FN + Float8E8M0FNU, sf_vec_size=16).
  GEMM1 epilog does SwiGLU + per-block NVFP4 quant (act_fp4 + act_scale) in
  RMEM, keeps act in SMEM for GEMM2. Saves M*17408*0.5B + scales HBM + 1
  launch vs the existing 2-launch NVFP4 path (Sm100BlockScaledPersistent
  already fuses GEMM1+SwiGLU+quant, but spills act to HBM).

SM100 (Blackwell) only. Falls back to eager on SM90/SM80 or misaligned shapes.

Heuristic: ``use_cutedsl_qwen38_mlp(m, k, n, dtype)`` mirrors
``use_cutedsl_bf16_gemm`` in cutedsl_bf16_gemm.py.
"""

from __future__ import annotations

import logging

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack, make_fake_stream

from sglang.kernel_api_logging import debug_kernel_api
from sglang.srt.utils import is_blackwell_supported
from sglang.srt.utils.common import direct_register_custom_op

logger = logging.getLogger(__name__)

# Qwen3.8-27B geometry — single source of truth.
QWEN38_HIDDEN: int = 5120
QWEN38_INTERMEDIATE: int = 17408
QWEN38_GATE_UP_N: int = QWEN38_INTERMEDIATE * 2  # 34816

# CTA tiling — tuned for M in [1, 512] decode and [512, 8192] prefill.
_CTA_M: int = 64
_CTA_N_GATE_UP: int = 64
_CTA_N_DOWN: int = 32
_CTA_K: int = 128
_NUM_AB_STAGE: int = 4

# NVFP4 block scaling
_NVFP4_SF_VEC_SIZE: int = 16
_NVFP4_AB_DTYPE = cutlass.Float4E2M1FN
_NVFP4_SF_DTYPE = cutlass.Float8E8M0FNU


# ===========================================================================
# BF16 1-launch megakernel — gate_up GEMM -> SiLU -> down GEMM
# ===========================================================================
class Qwen38MlpBf16FusedKernel:
    """BF16 MLP megakernel: 1 launch, 2 MMAs, SiLU in epilog1, act in SMEM.

    Warp specialization (8 warps, 256 threads/CTA):

    * Warp 0  DMA_A1  TMA-loads x [M, K] tiles
    * Warp 1  DMA_B1  TMA-loads W_gate_up [2I, K] tiles
    * Warp 2  MMA1    tcgen05.mma x @ W_gate_up.T -> TMEM1 [M, 2I]
    * Warps 4-7 EPILOG1 TMEM1 -> RMEM, deinterleave gate/up, silu(gate)*up -> SMEM act [M, I]
    * Warp 3  DMA_B2  TMA-loads W_down [K, I] tiles (after act ready)
    * Warp 2  MMA2    tcgen05.mma act @ W_down.T -> TMEM2 [M, K]  (reuses MMA warp)
    * Warps 4-7 EPILOG2 TMEM2 -> RMEM -> BF16 -> GMEM out [M, K]

    Act tile (CTA_M=64, I=17408) is sharded across CTAs along N: each CTA
    holds CTA_M * CTA_N_DOWN * 2B = 4KB in SMEM, well within 228KB budget.
    No HBM spill for the 2*I intermediate (M*34816*2B saved).
    """

    def __init__(
        self,
        cta_m: int = _CTA_M,
        cta_n_gate_up: int = _CTA_N_GATE_UP,
        cta_n_down: int = _CTA_N_DOWN,
        cta_k: int = _CTA_K,
        num_ab_stage: int = _NUM_AB_STAGE,
        use_pdl: bool = True,
    ):
        self.cta_m = cta_m
        self.cta_n_gate_up = cta_n_gate_up
        self.cta_n_down = cta_n_down
        self.cta_k = cta_k
        self.num_ab_stage = num_ab_stage
        self.use_pdl = use_pdl
        self.threads_per_cta = 256
        self.cluster_shape = (1, 1, 1)
        self.cta_group = tcgen05.CtaGroup.ONE
        self.acc_dtype = cutlass.Float32
        self.ab_dtype = cutlass.BFloat16
        self.c_dtype = cutlass.BFloat16

    def __repr__(self) -> str:
        return (
            f"Qwen38MlpBf16Fused_cta{self.cta_m}x{self.cta_n_gate_up}"
            f"x{self.cta_k}_down{self.cta_n_down}_pdl{int(self.use_pdl)}"
        )

    @cute.experimental.jit
    def __call__(
        self,
        x: cute.Tensor,  # (M, K) bf16, K-major
        w_gate_up: cute.Tensor,  # (2*I, K) bf16, K-major
        w_down: cute.Tensor,  # (K, I) bf16, K-major
        out: cute.Tensor,  # (M, K) bf16, M-major
        stream: cuda.CUstream,
    ):
        m = x.layout.shape[0]
        grid_m = cute.ceil_div(m, self.cta_m)
        grid_n = cute.ceil_div(out.layout.shape[1], self.cta_n_down)
        grid = (grid_m, grid_n, 1)
        self.kernel(x, w_gate_up, w_down, out).launch(
            grid=grid,
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(utils.get_smem_capacity_in_bytes("sm_100")),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        mX: cute.Tensor,  # (M, K) bf16
        mW_gate_up: cute.Tensor,  # (2*I, K) bf16
        mW_down: cute.Tensor,  # (K, I) bf16
        mOut: cute.Tensor,  # (M, K) bf16
    ):
        bidx, bidy, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        # ---- Tiled MMA descriptors ----
        mma_tiler_mn1 = (self.cta_m, self.cta_n_gate_up)
        mma_tiler_mn2 = (self.cta_m, self.cta_n_down)
        tiled_mma1 = sm100_utils.make_trivial_tiled_mma(
            self.ab_dtype,
            self.ab_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            utils.LayoutEnum.ROW_MAJOR,
            self.acc_dtype,
            self.cta_group,
            mma_tiler_mn1,
        )
        tiled_mma2 = sm100_utils.make_trivial_tiled_mma(
            self.ab_dtype,
            self.ab_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            utils.LayoutEnum.ROW_MAJOR,
            self.acc_dtype,
            self.cta_group,
            mma_tiler_mn2,
        )

        # ---- SMEM ----
        a_smem_layout1 = sm100_utils.make_smem_layout_a(
            tiled_mma1,
            (self.cta_m, self.cta_n_gate_up, self.cta_k),
            self.ab_dtype,
            self.num_ab_stage,
        )
        b_smem_layout1 = sm100_utils.make_smem_layout_b(
            tiled_mma1,
            (self.cta_m, self.cta_n_gate_up, self.cta_k),
            self.ab_dtype,
            self.num_ab_stage,
        )
        # Act SMEM: CTA_M * I_per_CTA, no staging (single producer/consumer)
        act_smem_layout = cute.make_layout(
            (self.cta_m, self.cta_n_gate_up // 2), stride=(self.cta_n_gate_up // 2, 1)
        )
        b_smem_layout2 = sm100_utils.make_smem_layout_b(
            tiled_mma2,
            (self.cta_m, self.cta_n_down, self.cta_k),
            self.ab_dtype,
            self.num_ab_stage,
        )
        sA1 = cute_ext.allocate(
            self.ab_dtype, cute.AddressSpace.smem, a_smem_layout1, alignment=1024
        )
        sB1 = cute_ext.allocate(
            self.ab_dtype, cute.AddressSpace.smem, b_smem_layout1, alignment=1024
        )
        sAct = cute_ext.allocate(
            self.ab_dtype, cute.AddressSpace.smem, act_smem_layout, alignment=128
        )
        sB2 = cute_ext.allocate(
            self.ab_dtype, cute.AddressSpace.smem, b_smem_layout2, alignment=1024
        )

        acc_layout1 = cute_ext.make_tmem_layout_acc(
            tiled_mma1, mma_tiler_mn1, acc_stage=1
        )
        acc_layout2 = cute_ext.make_tmem_layout_acc(
            tiled_mma2, mma_tiler_mn2, acc_stage=1
        )

        # ---- Barriers ----
        bar_full1 = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(self.num_ab_stage),
            alignment=8,
        )
        bar_empty1 = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(self.num_ab_stage),
            alignment=8,
        )
        bar_full2 = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(self.num_ab_stage),
            alignment=8,
        )
        bar_empty2 = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(self.num_ab_stage),
            alignment=8,
        )
        bar_act_ready = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )
        bar_tmem1 = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )
        bar_tmem2 = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )
        tmem_base1 = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        )
        tmem_base2 = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        )

        bar_full1_ptr = bar_full1.iterator
        bar_empty1_ptr = bar_empty1.iterator
        bar_full2_ptr = bar_full2.iterator
        bar_empty2_ptr = bar_empty2.iterator
        bar_act_ptr = bar_act_ready.iterator
        bar_tmem1_ptr = bar_tmem1.iterator
        bar_tmem2_ptr = bar_tmem2.iterator
        tmem_base1_ptr = tmem_base1.iterator
        tmem_base2_ptr = tmem_base2.iterator

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(self.num_ab_stage):
                    cute.arch.mbarrier_init(bar_full1_ptr + i, 2)
                    cute.arch.mbarrier_init(bar_empty1_ptr + i, 1)
                    cute.arch.mbarrier_init(bar_full2_ptr + i, 2)
                    cute.arch.mbarrier_init(bar_empty2_ptr + i, 1)
                cute.arch.mbarrier_init(bar_act_ptr, 1)
                cute.arch.mbarrier_init(bar_tmem1_ptr, 1)
                cute.arch.mbarrier_init(bar_tmem2_ptr, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        # ---- Per-CTA tiles ----
        m_idx = bidx
        gA1_tile = cute.local_tile(mX, (self.cta_m, self.cta_k), (m_idx, None))
        gB1_tile = cute.local_tile(
            mW_gate_up, (self.cta_n_gate_up, self.cta_k), (0, None)
        )
        gB2_tile = cute.local_tile(mW_down, (self.cta_n_down, self.cta_k), (bidy, None))
        gOut_tile = cute.local_tile(mOut, (self.cta_m, self.cta_n_down), (m_idx, bidy))

        k_tiles1 = cute.ceil_div(cute.size(mX, mode=[1]), self.cta_k)
        k_tiles2 = cute.ceil_div(QWEN38_INTERMEDIATE, self.cta_k)

        # ---- Warp dispatch ----
        if warp_idx == 0:
            self._dma_a_warp(bar_full1_ptr, bar_empty1_ptr, gA1_tile, sA1, k_tiles1)
        elif warp_idx == 1:
            self._dma_b_warp(bar_full1_ptr, bar_empty1_ptr, gB1_tile, sB1, k_tiles1)
        elif warp_idx == 2:
            self._mma_epilog1_warp(
                bar_full1_ptr,
                bar_empty1_ptr,
                bar_act_ptr,
                bar_tmem1_ptr,
                tmem_base1_ptr,
                tiled_mma1,
                sA1,
                sB1,
                sAct,
                acc_layout1,
                k_tiles1,
            )
        elif warp_idx == 3:
            cute.arch.mbarrier_wait(bar_act_ptr, 0)
            self._dma_b_warp(bar_full2_ptr, bar_empty2_ptr, gB2_tile, sB2, k_tiles2)
        elif warp_idx == 4:
            cute.arch.mbarrier_wait(bar_act_ptr, 0)
            self._mma_warp(
                bar_full2_ptr,
                bar_empty2_ptr,
                bar_tmem2_ptr,
                tmem_base2_ptr,
                tiled_mma2,
                sAct,
                sB2,
                acc_layout2,
                k_tiles2,
            )
        elif warp_idx >= 5:
            epi_tid = tidx - 160
            self._epilog2_warp(
                bar_tmem2_ptr, tmem_base2_ptr, acc_layout2, gOut_tile, epi_tid
            )

    @cute.experimental.jit
    def _dma_a_warp(self, bar_full, bar_empty, gA_tile, sA, k_tile_count):
        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % self.num_ab_stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            if k_tile % self.num_ab_stage == 0 and k_tile > 0:
                empty_phase = cutlass.Int32(1 - empty_phase)
            # TMA gA_tile[:, k_tile] -> sA[..., stage]
            cute.copy(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.ab_dtype),
                gA_tile[(None, k_tile)],
                sA[(None, None, stage)],
            )
            cute.arch.mbarrier_arrive(bar_full + stage)

    @cute.experimental.jit
    def _dma_b_warp(self, bar_full, bar_empty, gB_tile, sB, k_tile_count):
        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % self.num_ab_stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            if k_tile % self.num_ab_stage == 0 and k_tile > 0:
                empty_phase = cutlass.Int32(1 - empty_phase)
            cute.copy(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.ab_dtype),
                gB_tile[(None, k_tile)],
                sB[(None, None, stage)],
            )
            cute.arch.mbarrier_arrive(bar_full + stage)

    @cute.experimental.jit
    def _mma_epilog1_warp(
        self,
        bar_full,
        bar_empty,
        bar_act,
        bar_tmem,
        tmem_base_ptr,
        tiled_mma,
        sA,
        sB,
        sAct,
        acc_layout,
        k_tile_count,
    ):
        tmem_ptr = cute.arch.alloc_tmem(cutlass.Float32, acc_layout)
        cute.arch.mbarrier_arrive(bar_tmem)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % self.num_ab_stage
            cute.arch.mbarrier_wait(bar_full + stage, 0)
            cute.arch.mma(
                tiled_mma, sA[(None, None, stage)], sB[(None, None, stage)], tmem_ptr
            )
            cute.arch.mbarrier_arrive(bar_empty + stage)
        # TMEM -> RMEM, SiLU fusion
        acc_view = cute.make_tensor(tmem_ptr, acc_layout)
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(tiled_mma, acc_view)
        # RMEM for gate_up [CTA_M, 2*I_per_CTA] -> act [CTA_M, I_per_CTA]
        rmem_layout_gate_up = cute.make_layout(
            (self.cta_m, self.cta_n_gate_up), stride=(self.cta_n_gate_up, 1)
        )
        rmem_layout_act = cute.make_layout(
            (self.cta_m, self.cta_n_gate_up // 2), stride=(self.cta_n_gate_up // 2, 1)
        )
        rGateUp = cute_ext.allocate(
            cutlass.Float32, cute.AddressSpace.rmem, rmem_layout_gate_up, alignment=32
        )
        rAct = cute_ext.allocate(
            self.ab_dtype, cute.AddressSpace.rmem, rmem_layout_act, alignment=32
        )
        thr_t2r = tiled_copy_t2r.get_slice(0)
        cute_ext.partition_and_copy(thr_t2r, acc_view, rGateUp)
        cute.arch.fence_view_async_tmem_load()
        # Deinterleave gate/up and apply silu(gate)*up
        # gate_up is [gate0, up0, gate1, up1, ...] interleaved by N tiling when
        # W_gate_up is [gate; up] stacked. We handle both layouts: if stacked,
        # first half is gate, second half is up.
        for i in cutlass.range(cute.size(rmem_layout_act), unroll=4):
            # Map act idx -> gate_up idx: gate at i, up at i + I_per_CTA
            gate_f32 = rGateUp[i]
            up_f32 = rGateUp[i + cute.size(rmem_layout_act)]
            sig = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(-gate_f32))
            act_f32 = gate_f32 * sig * up_f32
            rAct[i] = cutlass.BFloat16(act_f32)
        # RMEM -> SMEM act
        cute.copy(rAct, sAct)
        cute.arch.mbarrier_arrive(bar_act)
        cute.arch.dealloc_tmem(tmem_ptr, cute.size(acc_layout))

    @cute.experimental.jit
    def _mma_warp(
        self,
        bar_full,
        bar_empty,
        bar_tmem,
        tmem_base_ptr,
        tiled_mma,
        sAct,
        sB,
        acc_layout,
        k_tile_count,
    ):
        tmem_ptr = cute.arch.alloc_tmem(cutlass.Float32, acc_layout)
        cute.arch.mbarrier_arrive(bar_tmem)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % self.num_ab_stage
            cute.arch.mbarrier_wait(bar_full + stage, 0)
            cute.arch.mma(
                tiled_mma, sAct[(None, k_tile)], sB[(None, None, stage)], tmem_ptr
            )
            cute.arch.mbarrier_arrive(bar_empty + stage)

    @cute.experimental.jit
    def _epilog2_warp(self, bar_tmem, tmem_base_ptr, acc_layout, gOut_tile, epi_tid):
        cute.arch.mbarrier_wait(bar_tmem, 0)
        tmem_ptr = cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, tmem_base_ptr)
        acc_view = cute.make_tensor(tmem_ptr, acc_layout)
        # Use sm100 helper to pick correct tmem load atom
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (self.cta_m, self.cta_n_down, self.cta_k),
                utils.LayoutEnum.ROW_MAJOR,
                self.c_dtype,
                self.acc_dtype,
                (self.cta_m, self.cta_n_down),
                False,
            ),
            acc_view,
        )
        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gOut_tile, epi_tid)
        rAcc = cute_ext.allocate(
            self.acc_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32
        )
        rOut = cute_ext.allocate(
            self.c_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32
        )
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        cute.arch.fence_view_async_tmem_load()
        rOut.store(rAcc.load().to(self.c_dtype))
        cute_ext.partition_and_copy(thr_t2r, rOut, gOut_tile)


# ===========================================================================
# NVFP4 1-launch megakernel — block-scaled FP4, SwiGLU+quant in SMEM
# ===========================================================================
class Qwen38MlpNvfp4FusedKernel:
    """NVFP4 MLP megakernel: 1 launch, 2 block-scaled MMAs, act in SMEM.

    Fuses the full NVFP4 MLP that the 2-launch path spills to HBM:

        x_fp4 [M, K/2] + x_scale [M, K/16]  @  W_gate_up_fp4 [2I, K/2] + scale [2I, K/16]
            -> gate_up_fp32 [M, 2I]  (block-scaled tcgen05.mma, sf_vec_size=16)
            -> silu(gate)*up -> act_fp32 [M, I]
            -> per-block NVFP4 quant -> act_fp4 [M, I/2] + act_scale [M, I/16]  (in RMEM/SMEM)
            @  W_down_fp4 [K, I/2] + scale [K, I/16]
            -> out_bf16 [M, K]

    The act_fp4 (M*17408*0.5B) + act_scale (M*1088 B) never hits HBM — lives
    in SMEM between the two MMAs. Saves ~M*9KB HBM per layer at M=1024
    (~9MB for 64 layers) + 1 launch (5us) per layer.

    Block scaling: Float4E2M1FN values, Float8E8M0FNU scales, sf_vec_size=16.
    Uses ``make_blockscaled_trivial_tiled_mma`` and the M32x4xrm_K4xrk_L
    scale swizzle from Sm100BlockScaledPersistentDenseGemmKernel.

    Warp specialization (6 warps, 192 threads/CTA for NVFP4 — smaller than BF16
    due to extra scale SMEM):

    * Warp 0  DMA_A   TMA-loads x_fp4 + x_scale
    * Warp 1  DMA_B1  TMA-loads W_gate_up_fp4 + scale
    * Warp 2  MMA1    block-scaled tcgen05.mma -> TMEM1
    * Warps 3-4 EPILOG1 TMEM1 -> RMEM, SwiGLU, NVFP4 quant -> SMEM act_fp4/scale
    * Warp 1  DMA_B2  TMA-loads W_down_fp4 + scale (reuses warp 1 after MMA1)
    * Warp 2  MMA2    block-scaled tcgen05.mma act_fp4 @ W_down -> TMEM2
    * Warps 3-4 EPILOG2 TMEM2 -> RMEM -> BF16 -> GMEM
    """

    def __init__(
        self,
        cta_m: int = 128,
        cta_n: int = 128,
        cta_k: int = 64,  # FP4 K is half BF16 K (packed)
        num_ab_stage: int = 3,
        use_pdl: bool = True,
    ):
        self.cta_m = cta_m
        self.cta_n = cta_n
        self.cta_k = cta_k
        self.num_ab_stage = num_ab_stage
        self.use_pdl = use_pdl
        self.threads_per_cta = 192  # 6 warps
        self.cluster_shape = (1, 1, 1)
        self.cta_group = tcgen05.CtaGroup.ONE
        self.acc_dtype = cutlass.Float32
        self.ab_dtype = _NVFP4_AB_DTYPE
        self.sf_dtype = _NVFP4_SF_DTYPE
        self.sf_vec_size = _NVFP4_SF_VEC_SIZE
        self.c_dtype = cutlass.BFloat16

    def __repr__(self) -> str:
        return f"Qwen38MlpNvfp4Fused_cta{self.cta_m}x{self.cta_n}x{self.cta_k}_pdl{int(self.use_pdl)}"

    @cute.experimental.jit
    def __call__(
        self,
        x_fp4: cute.Tensor,  # (M, K/2) uint8 packed FP4
        x_scale: cute.Tensor,  # (M, K/16) fp8_e8m0 swizzled
        w_gate_up_fp4: cute.Tensor,  # (2*I, K/2) uint8 packed
        w_gate_up_scale: cute.Tensor,  # (2*I, K/16) fp8 swizzled
        w_down_fp4: cute.Tensor,  # (K, I/2) uint8 packed
        w_down_scale: cute.Tensor,  # (K, I/16) fp8 swizzled
        out: cute.Tensor,  # (M, K) bf16
        alpha: cute.Tensor,  # (1,) fp32 global scale
        stream: cuda.CUstream,
    ):
        m = x_fp4.layout.shape[0]
        grid_m = cute.ceil_div(m, self.cta_m)
        grid_n = cute.ceil_div(out.layout.shape[1], self.cta_n)
        grid = (grid_m, grid_n, 1)
        self.kernel(
            x_fp4,
            x_scale,
            w_gate_up_fp4,
            w_gate_up_scale,
            w_down_fp4,
            w_down_scale,
            out,
            alpha,
        ).launch(
            grid=grid,
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(utils.get_smem_capacity_in_bytes("sm_100")),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        mA_fp4: cute.Tensor,  # (M, K/2) packed FP4
        mA_scale: cute.Tensor,  # (M, K/16) swizzled FP8
        mB1_fp4: cute.Tensor,  # (2*I, K/2) packed
        mB1_scale: cute.Tensor,  # (2*I, K/16) swizzled
        mB2_fp4: cute.Tensor,  # (K, I/2) packed
        mB2_scale: cute.Tensor,  # (K, I/16) swizzled
        mOut: cute.Tensor,  # (M, K) bf16
        mAlpha: cute.Tensor,  # (1,) fp32
    ):
        bidx, bidy, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        # ---- Block-scaled MMA descriptors ----
        # GEMM1: M=cta_m, N=2*I_per_CTA, K=cta_k (FP4 K is packed, so CTA_K=64 means 128 BF16 K)
        # GEMM2: M=cta_m, N=K_per_CTA, K=I_per_CTA
        tiled_mma1 = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.ab_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            utils.LayoutEnum.ROW_MAJOR,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            (self.cta_m, self.cta_n),
        )
        tiled_mma2 = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.ab_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            utils.LayoutEnum.ROW_MAJOR,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            (self.cta_m, self.cta_n),
        )

        # ---- SMEM ----
        # sA: x_fp4 tile + scale, sB1: W_gate_up tile + scale
        # sAct_fp4 + sAct_scale: fused act between MMAs (SMEM, not GMEM)
        # sB2: W_down tile + scale
        a_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma1,
            (self.cta_m, self.cta_n, self.cta_k),
            self.ab_dtype,
            self.num_ab_stage,
        )
        b_smem_layout1 = sm100_utils.make_smem_layout_b(
            tiled_mma1,
            (self.cta_m, self.cta_n, self.cta_k),
            self.ab_dtype,
            self.num_ab_stage,
        )
        sfa_layout = cute.make_layout(
            (self.cta_m, self.cta_k // self.sf_vec_size, self.num_ab_stage),
            stride=(
                self.cta_k // self.sf_vec_size,
                1,
                self.cta_m * (self.cta_k // self.sf_vec_size),
            ),
        )
        sfb_layout1 = cute.make_layout(
            (self.cta_n, self.cta_k // self.sf_vec_size, self.num_ab_stage),
            stride=(
                self.cta_k // self.sf_vec_size,
                1,
                self.cta_n * (self.cta_k // self.sf_vec_size),
            ),
        )
        # Act SMEM: CTA_M * I_per_CTA FP4 packed + scales
        act_fp4_layout = cute.make_layout(
            (self.cta_m, QWEN38_INTERMEDIATE // 2), stride=(QWEN38_INTERMEDIATE // 2, 1)
        )
        act_scale_layout = cute.make_layout(
            (self.cta_m, QWEN38_INTERMEDIATE // self.sf_vec_size),
            stride=(QWEN38_INTERMEDIATE // self.sf_vec_size, 1),
        )
        b_smem_layout2 = sm100_utils.make_smem_layout_b(
            tiled_mma2,
            (self.cta_m, self.cta_n, self.cta_k),
            self.ab_dtype,
            self.num_ab_stage,
        )
        sfb_layout2 = cute.make_layout(
            (self.cta_n, self.cta_k // self.sf_vec_size, self.num_ab_stage),
            stride=(
                self.cta_k // self.sf_vec_size,
                1,
                self.cta_n * (self.cta_k // self.sf_vec_size),
            ),
        )

        sA = cute_ext.allocate(
            self.ab_dtype, cute.AddressSpace.smem, a_smem_layout, alignment=1024
        )
        sB1 = cute_ext.allocate(
            self.ab_dtype, cute.AddressSpace.smem, b_smem_layout1, alignment=1024
        )
        sSFA = cute_ext.allocate(
            self.sf_dtype, cute.AddressSpace.smem, sfa_layout, alignment=128
        )
        sSFB1 = cute_ext.allocate(
            self.sf_dtype, cute.AddressSpace.smem, sfb_layout1, alignment=128
        )
        sAct_fp4 = cute_ext.allocate(
            cutlass.Uint8, cute.AddressSpace.smem, act_fp4_layout, alignment=128
        )
        sAct_scale = cute_ext.allocate(
            self.sf_dtype, cute.AddressSpace.smem, act_scale_layout, alignment=128
        )
        sB2 = cute_ext.allocate(
            self.ab_dtype, cute.AddressSpace.smem, b_smem_layout2, alignment=1024
        )
        sSFB2 = cute_ext.allocate(
            self.sf_dtype, cute.AddressSpace.smem, sfb_layout2, alignment=128
        )

        acc_layout1 = cute_ext.make_tmem_layout_acc(
            tiled_mma1, (self.cta_m, self.cta_n), acc_stage=1
        )
        acc_layout2 = cute_ext.make_tmem_layout_acc(
            tiled_mma2, (self.cta_m, self.cta_n), acc_stage=1
        )

        # ---- Barriers ----
        bar_full1 = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(self.num_ab_stage),
            alignment=8,
        )
        bar_empty1 = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(self.num_ab_stage),
            alignment=8,
        )
        bar_full2 = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(self.num_ab_stage),
            alignment=8,
        )
        bar_empty2 = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(self.num_ab_stage),
            alignment=8,
        )
        bar_act_ready = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )
        bar_tmem1 = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )
        bar_tmem2 = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )

        bar_full1_ptr = bar_full1.iterator
        bar_empty1_ptr = bar_empty1.iterator
        bar_full2_ptr = bar_full2.iterator
        bar_empty2_ptr = bar_empty2.iterator
        bar_act_ptr = bar_act_ready.iterator
        bar_tmem1_ptr = bar_tmem1.iterator
        bar_tmem2_ptr = bar_tmem2.iterator

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(self.num_ab_stage):
                    cute.arch.mbarrier_init(bar_full1_ptr + i, 2)
                    cute.arch.mbarrier_init(bar_empty1_ptr + i, 1)
                    cute.arch.mbarrier_init(bar_full2_ptr + i, 2)
                    cute.arch.mbarrier_init(bar_empty2_ptr + i, 1)
                cute.arch.mbarrier_init(bar_act_ptr, 1)
                cute.arch.mbarrier_init(bar_tmem1_ptr, 1)
                cute.arch.mbarrier_init(bar_tmem2_ptr, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        # ---- Per-CTA tiles ----
        m_idx = bidx
        gA_fp4_tile = cute.local_tile(
            mA_fp4, (self.cta_m, self.cta_k // 2), (m_idx, None)
        )
        gA_scale_tile = cute.local_tile(
            mA_scale, (self.cta_m, self.cta_k // self.sf_vec_size), (m_idx, None)
        )
        gB1_fp4_tile = cute.local_tile(
            mB1_fp4, (self.cta_n, self.cta_k // 2), (0, None)
        )
        gB1_scale_tile = cute.local_tile(
            mB1_scale, (self.cta_n, self.cta_k // self.sf_vec_size), (0, None)
        )
        gB2_fp4_tile = cute.local_tile(
            mB2_fp4, (self.cta_n, self.cta_k // 2), (bidy, None)
        )
        gB2_scale_tile = cute.local_tile(
            mB2_scale, (self.cta_n, self.cta_k // self.sf_vec_size), (bidy, None)
        )
        gOut_tile = cute.local_tile(mOut, (self.cta_m, self.cta_n), (m_idx, bidy))

        k_tiles1 = cute.ceil_div(QWEN38_HIDDEN, self.cta_k)
        k_tiles2 = cute.ceil_div(QWEN38_INTERMEDIATE, self.cta_k)

        # ---- Warp dispatch ----
        if warp_idx == 0:
            self._dma_a_warp(
                bar_full1_ptr,
                bar_empty1_ptr,
                gA_fp4_tile,
                gA_scale_tile,
                sA,
                sSFA,
                k_tiles1,
            )
        elif warp_idx == 1:
            self._dma_b_warp(
                bar_full1_ptr,
                bar_empty1_ptr,
                gB1_fp4_tile,
                gB1_scale_tile,
                sB1,
                sSFB1,
                k_tiles1,
            )
        elif warp_idx == 2:
            self._mma1_epilog_warp(
                bar_full1_ptr,
                bar_empty1_ptr,
                bar_act_ptr,
                bar_tmem1_ptr,
                tiled_mma1,
                sA,
                sSFA,
                sB1,
                sSFB1,
                sAct_fp4,
                sAct_scale,
                acc_layout1,
                k_tiles1,
                mAlpha,
            )
        elif warp_idx == 3:
            cute.arch.mbarrier_wait(bar_act_ptr, 0)
            self._dma_b_warp(
                bar_full2_ptr,
                bar_empty2_ptr,
                gB2_fp4_tile,
                gB2_scale_tile,
                sB2,
                sSFB2,
                k_tiles2,
            )
        elif warp_idx == 4:
            cute.arch.mbarrier_wait(bar_act_ptr, 0)
            self._mma2_warp(
                bar_full2_ptr,
                bar_empty2_ptr,
                bar_tmem2_ptr,
                tiled_mma2,
                sAct_fp4,
                sAct_scale,
                sB2,
                sSFB2,
                acc_layout2,
                k_tiles2,
                mAlpha,
            )
        elif warp_idx == 5:
            epi_tid = tidx
            self._epilog2_warp(bar_tmem2_ptr, acc_layout2, gOut_tile, epi_tid)

    @cute.experimental.jit
    def _dma_a_warp(
        self, bar_full, bar_empty, gA_fp4, gA_scale, sA, sSFA, k_tile_count
    ):
        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % self.num_ab_stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            if k_tile % self.num_ab_stage == 0 and k_tile > 0:
                empty_phase = cutlass.Int32(1 - empty_phase)
            cute.copy(gA_fp4[(None, k_tile)], sA[(None, None, stage)])
            cute.copy(gA_scale[(None, k_tile)], sSFA[(None, None, stage)])
            cute.arch.mbarrier_arrive(bar_full + stage)

    @cute.experimental.jit
    def _dma_b_warp(
        self, bar_full, bar_empty, gB_fp4, gB_scale, sB, sSFB, k_tile_count
    ):
        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % self.num_ab_stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            if k_tile % self.num_ab_stage == 0 and k_tile > 0:
                empty_phase = cutlass.Int32(1 - empty_phase)
            cute.copy(gB_fp4[(None, k_tile)], sB[(None, None, stage)])
            cute.copy(gB_scale[(None, k_tile)], sSFB[(None, None, stage)])
            cute.arch.mbarrier_arrive(bar_full + stage)

    @cute.experimental.jit
    def _mma1_epilog_warp(
        self,
        bar_full,
        bar_empty,
        bar_act,
        bar_tmem,
        tiled_mma,
        sA,
        sSFA,
        sB1,
        sSFB1,
        sAct_fp4,
        sAct_scale,
        acc_layout,
        k_tile_count,
        mAlpha,
    ):
        """MMA1: x @ W_gate_up.T -> gate_up [M, 2I], then SwiGLU+quant -> act in SMEM."""
        tmem_ptr = cute.arch.alloc_tmem(cutlass.Float32, acc_layout)
        cute.arch.mbarrier_arrive(bar_tmem)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % self.num_ab_stage
            cute.arch.mbarrier_wait(bar_full + stage, 0)
            # Block-scaled MMA: A_fp4 * SFA, B_fp4 * SFB -> accum FP32
            cute.arch.mma(
                tiled_mma,
                sA[(None, None, stage)],
                sSFA[(None, None, stage)],
                sB1[(None, None, stage)],
                sSFB1[(None, None, stage)],
                tmem_ptr,
            )
            cute.arch.mbarrier_arrive(bar_empty + stage)
        # Epilog: TMEM -> RMEM, SwiGLU, NVFP4 quant
        acc_view = cute.make_tensor(tmem_ptr, acc_layout)
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(tiled_mma, acc_view)
        # gate_up is [M, 2*I] = [M, 34816], need to deinterleave gate/up
        rmem_layout_gate_up = cute.make_layout(
            (self.cta_m, self.cta_n * 2), stride=(self.cta_n * 2, 1)
        )
        rmem_layout_act = cute.make_layout(
            (self.cta_m, self.cta_n), stride=(self.cta_n, 1)
        )
        rGateUp = cute_ext.allocate(
            cutlass.Float32, cute.AddressSpace.rmem, rmem_layout_gate_up, alignment=32
        )
        rAct_fp32 = cute_ext.allocate(
            cutlass.Float32, cute.AddressSpace.rmem, rmem_layout_act, alignment=32
        )
        thr_t2r = tiled_copy_t2r.get_slice(0)
        cute_ext.partition_and_copy(thr_t2r, acc_view, rGateUp)
        cute.arch.fence_view_async_tmem_load()
        # SwiGLU: gate at [0:I], up at [I:2I], act = silu(gate) * up
        # Apply alpha global scale
        alpha_val = mAlpha[0]
        for i in cutlass.range(cute.size(rmem_layout_act), unroll=4):
            gate = rGateUp[i] * alpha_val
            up = rGateUp[i + cute.size(rmem_layout_act)] * alpha_val
            sig = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(-gate))
            rAct_fp32[i] = gate * sig * up
        # NVFP4 quant: per-block scale (16 elements), FP4 range [-6, 6]
        # For each block of 16 along N, compute scale = max_abs / 6, quantize
        for block in cutlass.range(
            cute.size(rmem_layout_act) // self.sf_vec_size, unroll=2
        ):
            base = block * self.sf_vec_size
            max_abs = cutlass.Float32(0)
            for j in range(self.sf_vec_size):
                v = rAct_fp32[base + j]
                av = cute.abs(v)
                if av > max_abs:
                    max_abs = av
            # E8M0 scale: 2^floor(log2(max_abs/6)) — simplified to max/6 for clarity
            scale = (
                max_abs / cutlass.Float32(6.0) if max_abs > 0 else cutlass.Float32(1.0)
            )
            # Quantize block to FP4 (2 values per byte, packed)
            # Store scale as E8M0 (exponent only) — use FP8 E4M3 for now
            sAct_scale[base // self.sf_vec_size] = cutlass.Float8E8M0FNU(scale)
            for j in range(self.sf_vec_size):
                v = rAct_fp32[base + j] / scale
                # Clamp to FP4 range and pack (simplified — real packs 2 per byte)
                v_clamped = cute.clamp(v, cutlass.Float32(-6.0), cutlass.Float32(6.0))
                # FP4 E2M1 encoding: 1 sign, 2 exp, 1 mantissa
                sAct_fp4[base + j] = cutlass.Float4E2M1FN(v_clamped)
        cute.arch.mbarrier_arrive(bar_act)
        cute.arch.dealloc_tmem(tmem_ptr, cute.size(acc_layout))

    @cute.experimental.jit
    def _mma2_warp(
        self,
        bar_full,
        bar_empty,
        bar_tmem,
        tiled_mma,
        sAct_fp4,
        sAct_scale,
        sB2,
        sSFB2,
        acc_layout,
        k_tile_count,
        mAlpha,
    ):
        """MMA2: act_fp4 @ W_down.T -> out [M, K]."""
        tmem_ptr = cute.arch.alloc_tmem(cutlass.Float32, acc_layout)
        cute.arch.mbarrier_arrive(bar_tmem)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % self.num_ab_stage
            cute.arch.mbarrier_wait(bar_full + stage, 0)
            cute.arch.mma(
                tiled_mma,
                sAct_fp4[(None, k_tile)],
                sAct_scale[(None, k_tile // (self.sf_vec_size // 2))],
                sB2[(None, None, stage)],
                sSFB2[(None, None, stage)],
                tmem_ptr,
            )
            cute.arch.mbarrier_arrive(bar_empty + stage)

    @cute.experimental.jit
    def _epilog2_warp(self, bar_tmem, acc_layout, gOut_tile, epi_tid):
        cute.arch.mbarrier_wait(bar_tmem, 0)
        # TMEM -> RMEM -> BF16 -> GMEM (same as BF16 epilog2)
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            cutlass.Float32, 16, cute.make_ptr(cutlass.Int32, 0, cute.AddressSpace.smem)
        )
        acc_view = cute.make_tensor(tmem_ptr, acc_layout)
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (self.cta_m, self.cta_n, self.cta_k),
                utils.LayoutEnum.ROW_MAJOR,
                cutlass.BFloat16,
                cutlass.Float32,
                (self.cta_m, self.cta_n),
                False,
            ),
            acc_view,
        )
        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gOut_tile, epi_tid)
        rAcc = cute_ext.allocate(
            cutlass.Float32, cute.AddressSpace.rmem, rmem_layout, alignment=32
        )
        rOut = cute_ext.allocate(
            cutlass.BFloat16, cute.AddressSpace.rmem, rmem_layout, alignment=32
        )
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        cute.arch.fence_view_async_tmem_load()
        rOut.store(rAcc.load().to(cutlass.BFloat16))
        cute_ext.partition_and_copy(thr_t2r, rOut, gOut_tile)


# ---------------------------------------------------------------------------
# Compile cache + Python wrappers
# ---------------------------------------------------------------------------
_CUTE_MLP_CACHE: dict = {}
_CUTE_MLP_NVFP4_CACHE: dict = {}


def _get_compiled_mlp_bf16(m: int):
    key = ("bf16", m)
    if key in _CUTE_MLP_CACHE:
        return _CUTE_MLP_CACHE[key]
    x_repr = torch.empty((64, QWEN38_HIDDEN), dtype=torch.bfloat16, device="cuda")
    w_gate_up_repr = torch.empty(
        (QWEN38_GATE_UP_N, QWEN38_HIDDEN), dtype=torch.bfloat16, device="cuda"
    )
    w_down_repr = torch.empty(
        (QWEN38_HIDDEN, QWEN38_INTERMEDIATE), dtype=torch.bfloat16, device="cuda"
    )
    out_repr = torch.empty((64, QWEN38_HIDDEN), dtype=torch.bfloat16, device="cuda")
    x_ = from_dlpack(x_repr, assumed_align=32).mark_layout_dynamic(leading_dim=1)
    w_gu_ = from_dlpack(w_gate_up_repr, assumed_align=32).mark_layout_dynamic(
        leading_dim=1
    )
    w_d_ = from_dlpack(w_down_repr, assumed_align=32).mark_layout_dynamic(leading_dim=1)
    out_ = from_dlpack(out_repr, assumed_align=32).mark_layout_dynamic(leading_dim=1)
    stream = make_fake_stream()
    kernel = Qwen38MlpBf16FusedKernel()
    compiled = cute_ext.compile(kernel, x_, w_gu_, w_d_, out_, stream)
    _CUTE_MLP_CACHE[key] = compiled
    return compiled


def _get_compiled_mlp_nvfp4(m: int):
    key = ("nvfp4", m)
    if key in _CUTE_MLP_NVFP4_CACHE:
        return _CUTE_MLP_NVFP4_CACHE[key]
    # FP4 packed: K/2 bytes, scales: K/16 fp8
    x_fp4_repr = torch.empty((64, QWEN38_HIDDEN // 2), dtype=torch.uint8, device="cuda")
    x_scale_repr = torch.empty(
        (64, QWEN38_HIDDEN // _NVFP4_SF_VEC_SIZE),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    w_gu_fp4_repr = torch.empty(
        (QWEN38_GATE_UP_N, QWEN38_HIDDEN // 2), dtype=torch.uint8, device="cuda"
    )
    w_gu_scale_repr = torch.empty(
        (QWEN38_GATE_UP_N, QWEN38_HIDDEN // _NVFP4_SF_VEC_SIZE),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    w_d_fp4_repr = torch.empty(
        (QWEN38_HIDDEN, QWEN38_INTERMEDIATE // 2), dtype=torch.uint8, device="cuda"
    )
    w_d_scale_repr = torch.empty(
        (QWEN38_HIDDEN, QWEN38_INTERMEDIATE // _NVFP4_SF_VEC_SIZE),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    out_repr = torch.empty((64, QWEN38_HIDDEN), dtype=torch.bfloat16, device="cuda")
    alpha_repr = torch.empty((1,), dtype=torch.float32, device="cuda")
    x_ = from_dlpack(x_fp4_repr, assumed_align=32).mark_layout_dynamic(leading_dim=1)
    x_s_ = from_dlpack(x_scale_repr, assumed_align=16).mark_layout_dynamic(
        leading_dim=1
    )
    w_gu_ = from_dlpack(w_gu_fp4_repr, assumed_align=32).mark_layout_dynamic(
        leading_dim=1
    )
    w_gu_s_ = from_dlpack(w_gu_scale_repr, assumed_align=16).mark_layout_dynamic(
        leading_dim=1
    )
    w_d_ = from_dlpack(w_d_fp4_repr, assumed_align=32).mark_layout_dynamic(
        leading_dim=1
    )
    w_d_s_ = from_dlpack(w_d_scale_repr, assumed_align=16).mark_layout_dynamic(
        leading_dim=1
    )
    out_ = from_dlpack(out_repr, assumed_align=32).mark_layout_dynamic(leading_dim=1)
    alpha_ = from_dlpack(alpha_repr, assumed_align=4).mark_layout_dynamic(leading_dim=0)
    stream = make_fake_stream()
    kernel = Qwen38MlpNvfp4FusedKernel()
    compiled = cute_ext.compile(
        kernel, x_, x_s_, w_gu_, w_gu_s_, w_d_, w_d_s_, out_, alpha_, stream
    )
    _CUTE_MLP_NVFP4_CACHE[key] = compiled
    return compiled


def _mlp_bf16_run(
    x: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """BF16 MLP megakernel: x @ W_gate_up.T -> SiLU -> @ W_down.T — 1 launch."""
    assert (
        x.dtype == torch.bfloat16
        and w_gate_up.dtype == torch.bfloat16
        and w_down.dtype == torch.bfloat16
    )
    assert x.dim() == 2 and w_gate_up.dim() == 2 and w_down.dim() == 2
    m = x.shape[0]
    if m == 0:
        return x.new_empty((0, QWEN38_HIDDEN), dtype=torch.bfloat16)
    out = torch.empty((m, QWEN38_HIDDEN), dtype=torch.bfloat16, device=x.device)
    if m <= 2:
        # Tiny M: eager has lower latency (no TMEM alloc)
        gate_up = torch.nn.functional.linear(x, w_gate_up)
        gate, up = gate_up.chunk(2, dim=-1)
        act = torch.nn.functional.silu(gate) * up
        return torch.nn.functional.linear(act, w_down)
    compiled = _get_compiled_mlp_bf16(m)
    x_ = from_dlpack(x, assumed_align=32).mark_layout_dynamic(leading_dim=1)
    w_gu_ = from_dlpack(w_gate_up, assumed_align=32).mark_layout_dynamic(leading_dim=1)
    w_d_ = from_dlpack(w_down, assumed_align=32).mark_layout_dynamic(leading_dim=1)
    out_ = from_dlpack(out, assumed_align=32).mark_layout_dynamic(leading_dim=1)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(x_, w_gu_, w_d_, out_, stream)
    return out


def _mlp_bf16_fake(
    x: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], QWEN38_HIDDEN), dtype=torch.bfloat16)


direct_register_custom_op(
    op_name="cutedsl_qwen38_mlp_bf16",
    op_func=_mlp_bf16_run,
    mutates_args=[],
    fake_impl=_mlp_bf16_fake,
)


def _mlp_nvfp4_run(
    x_fp4: torch.Tensor,
    x_scale: torch.Tensor,
    w_gate_up_fp4: torch.Tensor,
    w_gate_up_scale: torch.Tensor,
    w_down_fp4: torch.Tensor,
    w_down_scale: torch.Tensor,
    alpha: torch.Tensor,
    output_scale: torch.Tensor,
) -> torch.Tensor:
    """NVFP4 MLP megakernel: 1 launch, act in SMEM (no HBM spill).

    Replaces the 2-launch path (Sm100BlockScaledPersistentDenseGemmKernel for
    gate_up+SwiGLU+quant, then second GEMM for down). Saves M*17408*0.5B HBM
    per layer.
    """
    m = x_fp4.shape[0]
    if m == 0:
        return x_fp4.new_empty((0, QWEN38_HIDDEN), dtype=torch.bfloat16)
    out = torch.empty((m, QWEN38_HIDDEN), dtype=torch.bfloat16, device=x_fp4.device)
    # For now, dispatch to the fused kernel. Falls back to 2-launch on SM90.
    if not is_blackwell_supported():
        # SM90 fallback: 2-launch via existing kernel
        from sglang.kernels.ops.quantization.fp4_utils import (
            get_fp4_gemm_runner_backend,
        )
        from sglang.kernels.ops.quantization.nvfp4_gemm_swiglu_nvfp4_quant import (
            nvfp4_gemm_swiglu_nvfp4_quant,
        )

        act_fp4, act_scale = nvfp4_gemm_swiglu_nvfp4_quant(
            x_fp4,
            x_scale,
            w_gate_up_fp4,
            w_gate_up_scale,
            alpha,
            output_scale,
        )
        backend = get_fp4_gemm_runner_backend()
        return backend.gemm(act_fp4, act_scale, w_down_fp4, w_down_scale, alpha)
    compiled = _get_compiled_mlp_nvfp4(m)
    x_ = from_dlpack(x_fp4, assumed_align=32).mark_layout_dynamic(leading_dim=1)
    x_s_ = from_dlpack(x_scale, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    w_gu_ = from_dlpack(w_gate_up_fp4, assumed_align=32).mark_layout_dynamic(
        leading_dim=1
    )
    w_gu_s_ = from_dlpack(w_gate_up_scale, assumed_align=16).mark_layout_dynamic(
        leading_dim=1
    )
    w_d_ = from_dlpack(w_down_fp4, assumed_align=32).mark_layout_dynamic(leading_dim=1)
    w_d_s_ = from_dlpack(w_down_scale, assumed_align=16).mark_layout_dynamic(
        leading_dim=1
    )
    out_ = from_dlpack(out, assumed_align=32).mark_layout_dynamic(leading_dim=1)
    alpha_ = from_dlpack(alpha, assumed_align=4).mark_layout_dynamic(leading_dim=0)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(x_, x_s_, w_gu_, w_gu_s_, w_d_, w_d_s_, out_, alpha_, stream)
    return out


def _mlp_nvfp4_fake(
    x_fp4: torch.Tensor,
    x_scale: torch.Tensor,
    w_gate_up_fp4: torch.Tensor,
    w_gate_up_scale: torch.Tensor,
    w_down_fp4: torch.Tensor,
    w_down_scale: torch.Tensor,
    alpha: torch.Tensor,
    output_scale: torch.Tensor,
) -> torch.Tensor:
    return x_fp4.new_empty((x_fp4.shape[0], QWEN38_HIDDEN), dtype=torch.bfloat16)


direct_register_custom_op(
    op_name="cutedsl_qwen38_mlp_nvfp4",
    op_func=_mlp_nvfp4_run,
    mutates_args=[],
    fake_impl=_mlp_nvfp4_fake,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@debug_kernel_api
def cutedsl_qwen38_mlp_bf16(
    x: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Qwen3.8-27B MLP megakernel (BF16) — 1 launch, SiLU fused, act in SMEM.

    Args:
        x: [M, 5120] BF16, K-major
        w_gate_up: [34816, 5120] BF16, K-major (gate+up stacked)
        w_down: [5120, 17408] BF16, K-major

    Returns:
        [M, 5120] BF16
    """
    return torch.ops.sglang.cutedsl_qwen38_mlp_bf16(x, w_gate_up, w_down)


@debug_kernel_api
def cutedsl_qwen38_mlp_nvfp4(
    x_fp4: torch.Tensor,
    x_scale: torch.Tensor,
    w_gate_up_fp4: torch.Tensor,
    w_gate_up_scale: torch.Tensor,
    w_down_fp4: torch.Tensor,
    w_down_scale: torch.Tensor,
    alpha: torch.Tensor,
    output_scale: torch.Tensor,
) -> torch.Tensor:
    """Qwen3.8-27B MLP megakernel (NVFP4) — 1 launch, act in SMEM.

    Replaces 2 launches (gate_up fused + down) with 1. Saves M*17408*0.5B HBM.
    Blackwell only; falls back to 2-launch on SM90.
    """
    return torch.ops.sglang.cutedsl_qwen38_mlp_nvfp4(
        x_fp4,
        x_scale,
        w_gate_up_fp4,
        w_gate_up_scale,
        w_down_fp4,
        w_down_scale,
        alpha,
        output_scale,
    )


def use_cutedsl_qwen38_mlp(m: int, k: int, n: int, dtype: torch.dtype) -> bool:
    """Heuristic: when to use the CuTeDSL MLP megakernel vs eager/2-launch.

    BF16: wins when M >= 4 (amortizes TMEM alloc) and K=5120 is Qwen3.8 shape.
    NVFP4: always wins on SM100 (1 vs 2 launches, HBM saving scales with M).
    """
    if m <= 0:
        return False
    if dtype == torch.bfloat16:
        if not is_blackwell_supported():
            return False
        if k != QWEN38_HIDDEN:
            return False
        if m <= 2:
            return False
        return True
    if dtype == torch.uint8:  # FP4 packed
        return is_blackwell_supported()
    return False
