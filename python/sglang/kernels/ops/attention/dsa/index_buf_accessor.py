from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.attention.dsa.utils import (
    INDEXER_K_CACHE_PRESHUFFLE_TILE,
    aiter_can_use_preshuffle_paged_mqa,
)
from sglang.srt.utils import get_bool_env_var, is_hip

_is_hip = is_hip()
_is_fp8_fnuz = is_fp8_fnuz()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
# aiter cp_gather kernel with preshuffle=True is only valid when the indexer
# uses the page_size=64 preshuffle layout (i.e. when the matching MQA gluon path
# is also enabled).
_use_aiter_preshuffle = aiter_can_use_preshuffle_paged_mqa()

if _use_aiter_preshuffle:
    from aiter.ops.cache import cp_gather_indexer_k_quant_cache

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

"""
k: data, 128 item per token, fp8
s: scale, 1 item per token, fp32
"""


class GetK:
    @classmethod
    def execute(cls, *args, **kwargs):
        return cls.triton(*args, **kwargs)

    @classmethod
    def slow(
        cls, pool: "DSATokenToKVPool", buf, seq_len: int, page_indices: torch.Tensor
    ):
        num_pages = (seq_len + pool.page_size - 1) // pool.page_size
        seq_len_ = num_pages * pool.page_size
        index_k_fp8 = torch.empty(
            (seq_len_, pool.index_head_dim),
            dtype=torch.uint8,
            device=pool.device,
        )
        for i in range(num_pages):
            page_index = page_indices[i]
            index_k_fp8[i * pool.page_size : (i + 1) * pool.page_size] = buf[
                page_index
            ][: pool.page_size * pool.index_head_dim].view(-1, pool.index_head_dim)

        return index_k_fp8[:seq_len]

    @classmethod
    def torch_fast(
        cls, pool: "DSATokenToKVPool", buf, seq_len: int, page_indices: torch.Tensor
    ):
        """
        :param page_indices: (num_pages,), int32
        :return: (seq_len, index_head_dim), uint8
        """

        # can handle per 128B instead of per element

        # page_indices: (num_pages,), element := a page index
        buf_numel_per_page = buf.shape[1]

        num_k_bytes_per_page = pool.page_size * pool.index_head_dim
        num_k_bytes_per_token = pool.index_head_dim

        # buf: (num_pages, page_size 64 * head_dim 128 + page_size 64 * fp32_nbytes 4), uint8
        # flat_buf: (whatever,), uint8
        flat_buf = buf.flatten()

        # flat_indices: (num_pages, num_k_bytes_per_page), int32, element := an index into flat_buf that we want to access
        flat_indices = (page_indices * buf_numel_per_page)[:, None] + torch.arange(
            num_k_bytes_per_page, dtype=torch.int32, device="cuda"
        )[None, :]
        flat_indices = flat_indices.flatten()[: seq_len * num_k_bytes_per_token]

        out = flat_buf[flat_indices]
        return out.view(-1, 128)

    @classmethod
    def triton(
        cls, pool: "DSATokenToKVPool", buf, seq_len: int, page_indices: torch.Tensor
    ):
        """
        Triton implementation for gathering K data from paged buffer.
        :param page_indices: (num_pages,), int32/int64
        :return: (seq_len, index_head_dim), uint8
        """
        return _get_k_triton(
            buf=buf,
            page_indices=page_indices,
            seq_len=seq_len,
            page_size=pool.page_size,
            index_head_dim=pool.index_head_dim,
        )


class GetS:
    @classmethod
    def execute(cls, *args, **kwargs):
        return cls.triton(*args, **kwargs)

    @classmethod
    def slow(
        cls, pool: "DSATokenToKVPool", buf, seq_len: int, page_indices: torch.Tensor
    ):
        num_pages = (seq_len + pool.page_size - 1) // pool.page_size
        seq_len_ = num_pages * pool.page_size
        assert pool.index_head_dim // pool.quant_block_size == 1
        index_k_scale_fp8 = torch.empty(
            (seq_len_, 4),
            dtype=torch.uint8,
            device=pool.device,
        )
        for i in range(num_pages):
            page_index = page_indices[i]
            index_k_scale_fp8[i * pool.page_size : (i + 1) * pool.page_size] = buf[
                page_index
            ][pool.page_size * pool.index_head_dim :].view(-1, 4)
        return index_k_scale_fp8[:seq_len]

    @classmethod
    def torch_fast(
        cls, pool: "DSATokenToKVPool", buf, seq_len: int, page_indices: torch.Tensor
    ):
        """
        :param page_indices: (num_pages,), int32
        :return: (seq_len, index_head_dim // quant_block_size), uint8
        """
        buf_numel_per_page = buf.shape[1]

        num_s_bytes_per_page = buf.shape[1] - pool.page_size * pool.index_head_dim
        num_s_bytes_per_token = pool.index_head_dim // pool.quant_block_size * 4
        s_offset_in_page = pool.page_size * pool.index_head_dim

        flat_buf = buf.flatten()
        flat_indices = (
            (page_indices * buf_numel_per_page)[:, None]
            + torch.arange(num_s_bytes_per_page, dtype=torch.int32, device="cuda")[
                None, :
            ]
            + s_offset_in_page
        )
        flat_indices = flat_indices.flatten()[: seq_len * num_s_bytes_per_token]

        out = flat_buf[flat_indices]
        return out.view(-1, 4)

    @classmethod
    def triton(
        cls, pool: "DSATokenToKVPool", buf, seq_len: int, page_indices: torch.Tensor
    ):
        """
        Triton implementation for gathering S (scale) data from paged buffer.
        :param page_indices: (num_pages,), int32/int64
        :return: (seq_len, 4), uint8
        """
        return _get_s_triton(
            buf=buf,
            page_indices=page_indices,
            seq_len=seq_len,
            page_size=pool.page_size,
            index_head_dim=pool.index_head_dim,
        )


class GetKAndS:
    @classmethod
    def execute(cls, *args, **kwargs):
        # The aiter path uses cp_gather_indexer_k_quant_cache(preshuffle=True),
        # which only matches the layout produced when the rest of the indexer
        # is on the page_size=64 preshuffle path. Otherwise fall back to the
        # triton implementation (which works on the page_size=1 legacy layout).
        if kwargs.get("integrity_args") is not None:
            return cls.triton(*args, **kwargs)
        if _use_aiter_preshuffle:
            return cls.aiter(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def aiter(
        cls,
        pool: "DSATokenToKVPool",
        buf: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_tensor: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ):
        from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype

        page_size = pool.page_size
        index_head_dim = pool.index_head_dim
        quant_block_size = pool.quant_block_size
        scale_elems = index_head_dim // quant_block_size

        kv_cache = buf.view(-1, page_size, index_head_dim + scale_elems * 4).view(
            fp8_dtype
        )
        dst_k = torch.empty(
            (seq_len_sum, index_head_dim), dtype=torch.uint8, device=buf.device
        )
        dst_scale = torch.empty(
            (seq_len_sum, scale_elems * 4), dtype=torch.uint8, device=buf.device
        )

        cu_seq_lens = torch.zeros(
            seq_len_tensor.shape[0] + 1, dtype=torch.int32, device=buf.device
        )
        torch.cumsum(seq_len_tensor.to(torch.int32), dim=0, out=cu_seq_lens[1:])

        cp_gather_indexer_k_quant_cache(
            kv_cache,
            dst_k.view(fp8_dtype),
            dst_scale,
            page_indices.to(torch.int32),
            cu_seq_lens,
            preshuffle=True,
        )
        return dst_k, dst_scale

    @classmethod
    def triton(
        cls,
        pool: "DSATokenToKVPool",
        buf: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_tensor: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
        integrity_args=None,
    ):
        """
        Triton implementation for gathering both K and S data from paged buffer in a single call.
        :param page_indices: (num_pages,), int32/int64
        :param seq_len_tensor: (num_pages,), int32/int64
        :param seq_len_sum: sum of all sequence len, int32
        :param max_seq_len: max of all sequence len, int32
        :return: tuple of (k_fp8, k_scale) where
                 k_fp8: (seq_len, index_head_dim), uint8
                 k_scale: (seq_len, 4), uint8
        """
        return _get_k_and_s_triton(
            buf=buf,
            page_indices=page_indices,
            seq_lens=seq_len_tensor,
            seq_len_sum=seq_len_sum,
            max_seq_len=max_seq_len,
            page_size=pool.page_size,
            index_head_dim=pool.index_head_dim,
            integrity_args=integrity_args,
        )


class SetKAndS:
    @classmethod
    def execute(cls, *args, buf, **kwargs):
        cls.triton(*args, **kwargs, buf=buf)

    @classmethod
    def triton(cls, pool, buf, loc, index_k, index_k_scale):
        loc = loc.to(torch.int64)

        _set_k_and_s_triton(
            buf=buf,
            loc=loc,
            index_k=index_k,
            index_k_scale=index_k_scale,
            page_size=pool.page_size,
        )


def _set_k_and_s_triton(
    buf: torch.Tensor,
    loc: torch.Tensor,
    index_k: torch.Tensor,
    index_k_scale: torch.Tensor,
    page_size: int,
):
    """
    :param buf: (num_pages, page_size 64 * (128B data + 4B scale)), uint8
    :param loc: (num_tokens_to_write,), int, element := the token index to write to
    :param index_k: (num_tokens_to_write, 128 elem), fp8
    :param index_k_scale: (num_tokens_to_write, 1 elem), fp32
    :return:
    """
    num_pages, buf_numel_per_page = buf.shape
    (num_tokens_to_write,) = loc.shape
    num_tokens_to_write_, index_head_dim = index_k.shape

    # Handle both 1D (num_tokens,) and 2D (num_tokens, 1) shapes for index_k_scale
    if index_k_scale.ndim == 1:
        num_tokens_to_write__ = index_k_scale.shape[0]
        scale_dim = 1
    elif index_k_scale.ndim == 2:
        num_tokens_to_write__, scale_dim = index_k_scale.shape
    else:
        raise ValueError(
            f"index_k_scale must be 1D or 2D, got shape {index_k_scale.shape}"
        )
    assert buf_numel_per_page == page_size * (128 + 4)
    assert num_tokens_to_write == num_tokens_to_write_ == num_tokens_to_write__
    assert index_head_dim == 128
    assert scale_dim == 1
    if _is_hip:
        if _use_aiter_preshuffle:
            assert (
                page_size % 16 == 0
            ), f"HIP preshuffle requires page_size to be a multiple of 16, got {page_size}"
    else:
        assert page_size == 64

    assert buf.dtype == torch.uint8
    assert loc.dtype == torch.int64, f"{loc.dtype=}"  # can be int32
    if _is_fp8_fnuz:
        assert index_k.dtype == torch.float8_e4m3fnuz
    else:
        assert index_k.dtype == torch.float8_e4m3fn
    assert index_k_scale.dtype == torch.float32

    assert buf.is_contiguous()
    assert loc.is_contiguous()
    assert index_k.is_contiguous()
    assert index_k_scale.is_contiguous()

    if _is_fp8_fnuz:
        buf_fp8 = buf.view(torch.float8_e4m3fnuz)
    else:
        buf_fp8 = buf.view(torch.float8_e4m3fn)
    buf_fp32 = buf.view(torch.float32)

    _set_k_and_s_triton_kernel[(num_tokens_to_write,)](
        buf_fp8,
        buf_fp32,
        loc,
        index_k,
        index_k_scale,
        index_k.stride(0),
        PAGE_SIZE=page_size,
        BUF_NUMEL_PER_PAGE=buf_numel_per_page,
        NUM_K_ELEMS_PER_TOKEN=index_head_dim,
        S_OFFSET_NBYTES_IN_PAGE=page_size * index_head_dim,
        PRESHUFFLE_TILE=INDEXER_K_CACHE_PRESHUFFLE_TILE if _use_aiter_preshuffle else 0,
    )


@triton.jit
def _set_k_and_s_triton_kernel(
    buf_fp8_ptr,
    buf_fp32_ptr,
    loc_ptr,
    index_k_ptr,
    index_k_scale_ptr,
    index_k_ptr_stride_0,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    NUM_K_ELEMS_PER_TOKEN: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    PRESHUFFLE_TILE: tl.constexpr,
):
    token_id = tl.program_id(0)

    loc = tl.load(loc_ptr + token_id)

    in_k_offsets = token_id * index_k_ptr_stride_0 + tl.arange(0, NUM_K_ELEMS_PER_TOKEN)

    # no need for `mask`, since we read 128B for k and 4B for scale, both pow of 2
    k = tl.load(index_k_ptr + in_k_offsets)
    k_scale = tl.load(index_k_scale_ptr + token_id)

    loc_page_index = loc // PAGE_SIZE
    loc_token_offset_in_page = loc % PAGE_SIZE

    k_range = tl.arange(0, NUM_K_ELEMS_PER_TOKEN)
    if PRESHUFFLE_TILE:
        tile = PRESHUFFLE_TILE
        token_tile_id = loc_token_offset_in_page // tile
        token_in_tile = loc_token_offset_in_page % tile
        col_tile_id = k_range // tile
        col_in_tile = k_range % tile
        out_k_offsets = (
            loc_page_index * BUF_NUMEL_PER_PAGE
            + token_tile_id * (tile * NUM_K_ELEMS_PER_TOKEN)
            + col_tile_id * (tile * tile)
            + token_in_tile * tile
            + col_in_tile
        )
    else:
        out_k_offsets = (
            loc_page_index * BUF_NUMEL_PER_PAGE
            + loc_token_offset_in_page * NUM_K_ELEMS_PER_TOKEN
            + k_range
        )

    # "//4" b/c it is fp32 instead of uint8
    out_s_offset = (
        loc_page_index * BUF_NUMEL_PER_PAGE // 4
        + S_OFFSET_NBYTES_IN_PAGE // 4
        + loc_token_offset_in_page
    )

    tl.store(buf_fp8_ptr + out_k_offsets, k)
    tl.store(buf_fp32_ptr + out_s_offset, k_scale)


def _get_k_triton(
    buf: torch.Tensor,
    page_indices: torch.Tensor,
    seq_len: int,
    page_size: int,
    index_head_dim: int,
):
    """
    Gather K (key) data from paged buffer using Triton.

    :param buf: (num_pages, page_size * 128 + page_size * 4), uint8
    :param page_indices: (num_pages,), int32/int64
    :param seq_len: int, number of tokens to gather
    :param page_size: int, typically 64
    :param index_head_dim: int, typically 128
    :return: (seq_len, index_head_dim), uint8
    """
    num_pages, buf_numel_per_page = buf.shape

    # Allocate output
    out = torch.empty((seq_len, index_head_dim), dtype=torch.uint8, device=buf.device)

    # Launch kernel with one thread per token
    grid = (seq_len,)
    _get_k_triton_kernel[grid](
        buf,
        page_indices,
        out,
        seq_len,
        page_size,
        buf_numel_per_page,
        index_head_dim,
        BLOCK_SIZE=128,
    )

    return out


@triton.jit
def _get_k_triton_kernel(
    buf_ptr,
    page_indices_ptr,
    out_ptr,
    seq_len: tl.constexpr,
    page_size: tl.constexpr,
    buf_numel_per_page: tl.constexpr,
    index_head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program handles one token (seq_len tokens total).
    Loads 128 bytes from the appropriate page.
    """
    token_id = tl.program_id(0)

    # Calculate which page and offset within page
    page_idx = token_id // page_size
    token_offset_in_page = token_id % page_size

    # Load the page index from page_indices
    page_index = tl.load(page_indices_ptr + page_idx)

    # Calculate source offset in buf
    # buf[page_index, token_offset_in_page * index_head_dim : ...]
    src_base_offset = (
        page_index * buf_numel_per_page + token_offset_in_page * index_head_dim
    )

    # Load 128 bytes (index_head_dim elements)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < index_head_dim
    data = tl.load(buf_ptr + src_base_offset + offsets, mask=mask)

    # Store to output
    dst_offset = token_id * index_head_dim
    tl.store(out_ptr + dst_offset + offsets, data, mask=mask)


def _get_s_triton(
    buf: torch.Tensor,
    page_indices: torch.Tensor,
    seq_len: int,
    page_size: int,
    index_head_dim: int,
):
    """
    Gather S (scale) data from paged buffer using Triton.

    :param buf: (num_pages, page_size * 128 + page_size * 4), uint8
    :param page_indices: (num_pages,), int32/int64
    :param seq_len: int, number of tokens to gather
    :param page_size: int, typically 64
    :param index_head_dim: int, typically 128
    :return: (seq_len, 4), uint8 (representing fp32 scale)
    """
    num_pages, buf_numel_per_page = buf.shape
    s_offset_in_page = page_size * index_head_dim  # Scales start after K data

    # Allocate output
    out = torch.empty((seq_len, 4), dtype=torch.uint8, device=buf.device)

    # Launch kernel with one thread per token
    grid = (seq_len,)
    _get_s_triton_kernel[grid](
        buf,
        page_indices,
        out,
        seq_len,
        page_size,
        buf_numel_per_page,
        s_offset_in_page,
    )

    return out


@triton.jit
def _get_s_triton_kernel(
    buf_ptr,
    page_indices_ptr,
    out_ptr,
    seq_len: tl.constexpr,
    page_size: tl.constexpr,
    buf_numel_per_page: tl.constexpr,
    s_offset_in_page: tl.constexpr,
):
    """
    Each program handles one token (seq_len tokens total).
    Loads 4 bytes (fp32 scale) from the appropriate page.
    """
    token_id = tl.program_id(0)

    # Calculate which page and offset within page
    page_idx = token_id // page_size
    token_offset_in_page = token_id % page_size

    # Load the page index from page_indices
    page_index = tl.load(page_indices_ptr + page_idx)

    # Calculate source offset in buf
    # Scales are stored after K data: page_size * index_head_dim offset
    # buf[page_index, s_offset_in_page + token_offset_in_page * 4 : ...]
    src_base_offset = (
        page_index * buf_numel_per_page + s_offset_in_page + token_offset_in_page * 4
    )

    # Load 4 bytes (fp32 scale)
    offsets = tl.arange(0, 4)
    data = tl.load(buf_ptr + src_base_offset + offsets)

    # Store to output
    dst_offset = token_id * 4
    tl.store(out_ptr + dst_offset + offsets, data)


def _get_k_and_s_triton(
    buf: torch.Tensor,
    page_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_len_sum: int,
    max_seq_len: int,
    page_size: int,
    index_head_dim: int,
    integrity_args=None,
):
    """
    Fused gather of both K (key) and S (scale) data from paged buffer using Triton.
    This is more efficient than calling GetK and GetS separately.

    :param buf: (num_pages, page_size * 128 + page_size * 4), uint8
    :param page_indices: (num_pages,), int32/int64
    :param seq_lens: tensor of sequence lens, int64
    :param seq_len_sum: sum of all sequence len, int32
    :param max_seq_len: max of sequence len, int32
    :param page_size: int, typically 64
    :param index_head_dim: int, typically 128
    :return: tuple of (k_out, s_out) where
             k_out: (seq_len, index_head_dim), uint8
             s_out: (seq_len, 4), uint8
    """
    # Allocate outputs
    k_out = torch.empty(
        (seq_len_sum, index_head_dim), dtype=torch.uint8, device=buf.device
    )
    s_out = torch.empty((seq_len_sum, 4), dtype=torch.uint8, device=buf.device)

    _, buf_numel_per_page = buf.shape
    _, page_indice_batch_offset = page_indices.shape
    s_offset_in_page = page_size * index_head_dim

    # Launch kernel with one thread per token
    BLOCK_SIZE = 256
    BLOCK_SIZE_K = 128

    num_token_blocks = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_k_threads = (index_head_dim + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K

    seq_num = seq_lens.shape[0]
    grid = (seq_num, num_token_blocks, num_k_threads)
    seq_num_pow2 = 1
    while seq_num_pow2 < seq_num:
        seq_num_pow2 *= 2

    protect = integrity_args is not None
    if protect:
        request_indices = integrity_args["request_indices"]
        generations = integrity_args["generations"]
        request_epochs = integrity_args["request_epochs"]
        expected_tags = integrity_args["expected_tags"]
        failure_status = integrity_args["failure_status"]
        mapping_seed = integrity_args["mapping_seed"]
        if request_indices.numel() != seq_num:
            raise ValueError("protected K/S gather requires one request per sequence")
        if expected_tags.shape[0] != request_epochs.numel():
            raise ValueError("protected K/S gather request sidecar mismatch")
    else:
        request_indices = page_indices
        generations = page_indices
        request_epochs = page_indices
        expected_tags = page_indices
        failure_status = page_indices
        mapping_seed = 0

    _get_k_and_s_triton_kernel[grid](
        buf_ptr=buf,
        page_indices_ptr=page_indices,
        request_indices_ptr=request_indices,
        generations_ptr=generations,
        request_epochs_ptr=request_epochs,
        expected_tags_ptr=expected_tags,
        failure_status_ptr=failure_status,
        k_out_ptr=k_out,
        s_out_ptr=s_out,
        seq_len_ptr=seq_lens,
        seq_len_num_pow=seq_num_pow2,
        seq_len_sum=seq_len_sum,
        max_seq_len=max_seq_len,
        page_size=page_size,
        buf_numel_per_page=buf_numel_per_page,
        index_head_dim=index_head_dim,
        s_offset_in_page=s_offset_in_page,
        page_indice_batch_offset=page_indice_batch_offset,
        physical_capacity=generations.numel() if protect else 0,
        request_capacity=request_epochs.numel() if protect else 0,
        logical_capacity=expected_tags.shape[1] if protect else 0,
        mapping_seed=mapping_seed,
        PROTECT=protect,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return k_out, s_out


@triton.jit
def _get_k_and_s_triton_kernel(
    buf_ptr,
    page_indices_ptr,
    request_indices_ptr,
    generations_ptr,
    request_epochs_ptr,
    expected_tags_ptr,
    failure_status_ptr,
    k_out_ptr,
    s_out_ptr,
    seq_len_ptr,
    seq_len_num_pow: tl.constexpr,
    seq_len_sum,
    max_seq_len,
    page_size: tl.constexpr,
    buf_numel_per_page: tl.constexpr,
    index_head_dim: tl.constexpr,
    s_offset_in_page: tl.constexpr,
    page_indice_batch_offset,
    physical_capacity: tl.constexpr,
    request_capacity: tl.constexpr,
    logical_capacity: tl.constexpr,
    mapping_seed,
    PROTECT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused kernel that gathers both K and S data in a single pass.
    Each program handles one token (seq_len tokens total).
    Loads 128 bytes (K) + 4 bytes (S) from the appropriate page.
    """
    batch_id = tl.program_id(0)
    block_token_start = tl.program_id(1) * BLOCK_SIZE
    thread_idx = tl.program_id(2)

    # Define the token range within the block and the K dimension range handled by the thread.
    token_ids_in_block = tl.arange(0, BLOCK_SIZE)
    token_ids = block_token_start + token_ids_in_block
    k_offsets = thread_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    raw_seq_len = tl.load(seq_len_ptr + batch_id)
    seq_len = tl.maximum(0, tl.minimum(raw_seq_len, max_seq_len))
    # Grid axis 1 spans the batch-max seq len; fully-masked blocks store nothing.
    if block_token_start >= seq_len:
        return
    pre_batch_idx = tl.arange(0, seq_len_num_pow)
    mask_pre_batch_idx = pre_batch_idx < batch_id
    prev_seq_lens = tl.load(
        seq_len_ptr + pre_batch_idx, mask=mask_pre_batch_idx, other=0
    )
    prev_seq_lens = tl.maximum(0, tl.minimum(prev_seq_lens, max_seq_len))
    batch_token_offset = tl.sum(prev_seq_lens.to(tl.int64))
    dst_token_ids = batch_token_offset + token_ids.to(tl.int64)
    token_valid_mask = (
        (token_ids < seq_len)
        & (token_ids < max_seq_len)
        & (dst_token_ids >= 0)
        & (dst_token_ids < seq_len_sum)
    )

    # Batch calculate the page index and in-page offset of each token.
    page_idx = token_ids // page_size
    token_offset_in_page = token_ids % page_size
    page_indices_base = batch_id * page_indice_batch_offset
    page_idx_valid_mask = page_idx < page_indice_batch_offset
    page_index = tl.load(
        page_indices_ptr + page_idx + page_indices_base,
        mask=token_valid_mask & page_idx_valid_mask,
        other=0,
    )

    if PROTECT:
        request = tl.load(request_indices_ptr + batch_id)
        request_in_range = (request > 0) & (request < request_capacity)
        page_in_range = (page_index > 0) & (page_index < physical_capacity)
        logical_in_range = page_idx < logical_capacity
        safe_request = tl.where(request_in_range, request, 0)
        safe_page = tl.where(page_in_range, page_index, 0)
        safe_logical = tl.where(logical_in_range, page_idx, 0)
        generation = tl.load(generations_ptr + safe_page)
        request_epoch = tl.load(request_epochs_ptr + safe_request)
        expected_tag = tl.load(
            expected_tags_ptr + safe_request * logical_capacity + safe_logical
        ).to(tl.uint64)

        value = mapping_seed.to(tl.uint64) ^ 0x445356344D415032
        value ^= safe_request.to(tl.uint64) * 0x9E3779B97F4A7C15
        value ^= request_epoch.to(tl.uint64) * 0xBF58476D1CE4E5B9
        value ^= safe_logical.to(tl.uint64) * 0x94D049BB133111EB
        value ^= safe_page.to(tl.uint64) * 0xD6E8FEB86659FD93
        value ^= generation.to(tl.uint64) * 0xA0761D6478BD642F
        value ^= value >> 30
        value *= 0xBF58476D1CE4E5B9
        value ^= value >> 27
        value *= 0x94D049BB133111EB
        actual_tag = value ^ (value >> 31)
        actual_tag = tl.where(actual_tag == 0, 1, actual_tag)

        failure = tl.where(request_in_range, 0, 1 << 6)
        failure = tl.where(
            request_in_range & (~page_in_range | ~logical_in_range),
            1 << 0,
            failure,
        )
        valid_geometry = request_in_range & page_in_range & logical_in_range
        failure = tl.where(
            valid_geometry & ((generation == 0) | (request_epoch == 0)),
            1 << 5,
            failure,
        )
        live_identity = valid_geometry & (generation != 0) & (request_epoch != 0)
        failure = tl.where(live_identity & (expected_tag == 0), 1 << 3, failure)
        failure = tl.where(
            live_identity & (expected_tag != 0) & (expected_tag != actual_tag),
            1 << 4,
            failure,
        )
        mapping_valid = failure == 0
        report_failure = (
            token_valid_mask
            & page_idx_valid_mask
            & (token_offset_in_page == 0)
            & request_in_range
            & (failure != 0)
        )
        tl.atomic_or(
            failure_status_ptr + safe_request + tl.zeros_like(failure).to(tl.int64),
            failure.to(tl.int32),
            mask=report_failure,
        )
        page_index = tl.where(mapping_valid, page_index, 0)

    # ===== Load K data =====
    # The address calculation logic for K: page_index * total number of elements in a single page + K offset of the token within the page.
    k_src_token_offset = token_offset_in_page * index_head_dim
    k_src_base_offset = page_index * buf_numel_per_page + k_src_token_offset

    k_load_addr = buf_ptr + k_src_base_offset[:, None] + k_offsets[None, :]
    k_dim_mask = k_offsets[None, :] < index_head_dim
    k_mask = token_valid_mask[:, None] & k_dim_mask

    k_data = tl.load(k_load_addr, mask=k_mask, other=0)

    # Store K to output
    k_dst_token_offset = dst_token_ids
    k_dst_base_offset = k_dst_token_offset * index_head_dim
    k_store_addr = k_out_ptr + k_dst_base_offset[:, None] + k_offsets[None, :]
    tl.store(k_store_addr, k_data, mask=k_mask)

    # ===== Load S data =====
    # The address calculation logic for S: page_index * total number of elements in a single page + starting offset of S within the page + offset of token within S in the page
    s_src_token_offset = s_offset_in_page + token_offset_in_page * 4
    s_src_base_offset = page_index * buf_numel_per_page + s_src_token_offset

    s_offsets = tl.arange(0, 4)
    s_load_addr = buf_ptr + s_src_base_offset[:, None] + s_offsets[None, :]
    s_mask = token_valid_mask[:, None] & (s_offsets[None, :] < 4)
    s_data = tl.load(s_load_addr, mask=s_mask, other=0)

    # Store S to output
    s_dst_token_offset = dst_token_ids
    s_dst_base_offset = s_dst_token_offset * 4
    s_store_addr = s_out_ptr + s_dst_base_offset[:, None] + s_offsets[None, :]
    tl.store(s_store_addr, s_data, mask=s_mask)
