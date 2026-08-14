from __future__ import annotations

import pytest
import torch

from sglang.kernels.ops.attention.deepseek_v4_rope import (
    precompute_freqs_cis,
)
from sglang.kernels.ops.attention.dsv4 import (
    CompressorDecodePlan,
    compress_forward,
    compress_norm_rope_store,
)
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.deepseek_v4.common import (
    make_paged_context,
    make_state_pool,
)

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize("compress_ratio", [4, 128])
@torch.inference_mode()
def test_compressor_waits_before_reading_graph_updated_plan(
    compress_ratio: int,
) -> None:
    """A PDL consumer must not derive pointers from the previous plan value."""
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("programmatic dependent launch requires SM90 or newer")

    batch_size = 8
    head_dim = 512
    ctx = make_paged_context(
        bs=batch_size,
        compress_ratio=compress_ratio,
        head_dim=head_dim,
    )
    seq_lens = torch.full(
        (batch_size,), compress_ratio, dtype=torch.int64, device=get_device()
    )
    safe_plan = ctx.make_decode_plan(seq_lens)
    live_plan = CompressorDecodePlan(compress_ratio, safe_plan.plan_d.clone())
    invalid_plan = safe_plan.plan_d.clone()
    invalid_plan.view(torch.int32)[:, 1:].fill_(1 << 29)

    pool = make_state_pool(ctx.num_pages, compress_ratio, head_dim)
    input_width = head_dim * (4 if compress_ratio == 4 else 2)
    kv_input = torch.randn(
        batch_size,
        input_width,
        dtype=torch.float32,
        device=get_device(),
    )
    ape = torch.randn(
        compress_ratio * (2 if compress_ratio == 4 else 1),
        head_dim,
        dtype=torch.float32,
        device=get_device(),
    )
    output = torch.empty(batch_size, head_dim, dtype=torch.float32, device=get_device())

    # Compile the JIT module before capture.
    compress_forward(
        pool,
        kv_input,
        ape,
        live_plan,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
        out=output,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=capture_stream):
        live_plan.plan_d.copy_(safe_plan.plan_d)
        compress_forward(
            pool,
            kv_input,
            ape,
            live_plan,
            head_dim=head_dim,
            compress_ratio=compress_ratio,
            out=output,
        )

    # Poison the graph-static plan before every replay. The captured copy is
    # the primary launch; consumers that read before PDLWaitPrimary can observe
    # these values and form out-of-bounds state-buffer pointers.
    for _ in range(100):
        live_plan.plan_d.copy_(invalid_plan)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()

    assert torch.isfinite(output).all()


@pytest.mark.parametrize("compress_ratio", [4, 128])
@torch.inference_mode()
def test_zero_length_decode_row_is_inactive_for_compressor(
    compress_ratio: int,
) -> None:
    """CUDA-graph padding rows must not read or write compressor state."""
    batch_size = 2
    head_dim = 512
    ctx = make_paged_context(
        bs=batch_size,
        compress_ratio=compress_ratio,
        head_dim=head_dim,
    )
    seq_lens = torch.tensor([compress_ratio, 0], dtype=torch.int64, device=get_device())
    plan = ctx.make_decode_plan(seq_lens)
    raw_plan = plan.plan_d.view(torch.int32)
    torch.testing.assert_close(raw_plan[1], torch.zeros_like(raw_plan[1]))

    pool = make_state_pool(ctx.num_pages, compress_ratio, head_dim)
    zero_slot_before = pool[0, 0].clone()
    input_width = head_dim * (4 if compress_ratio == 4 else 2)
    kv_input = torch.randn(
        batch_size,
        input_width,
        dtype=torch.float32,
        device=get_device(),
    )
    ape = torch.randn(
        compress_ratio * (2 if compress_ratio == 4 else 1),
        head_dim,
        dtype=torch.float32,
        device=get_device(),
    )
    output = torch.full(
        (batch_size, head_dim),
        -123.0,
        dtype=torch.float32,
        device=get_device(),
    )

    compress_forward(
        pool,
        kv_input,
        ape,
        plan,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
        out=output,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output[1], torch.full_like(output[1], -123.0))
    torch.testing.assert_close(pool[0, 0], zero_slot_before)


@torch.inference_mode()
def test_zero_length_decode_row_is_inactive_for_fused_norm_rope() -> None:
    """A padded row must return before deriving a negative RoPE position."""
    compress_ratio = 4
    head_dim = 128
    ctx = make_paged_context(
        bs=2,
        compress_ratio=compress_ratio,
        head_dim=head_dim,
    )
    seq_lens = torch.tensor([compress_ratio, 0], dtype=torch.int64, device=get_device())
    plan = ctx.make_decode_plan(seq_lens)
    kv = torch.randn(2, head_dim, dtype=torch.bfloat16, device=get_device())
    norm_weight = torch.randn(head_dim, dtype=torch.bfloat16, device=get_device())
    freqs_cis = precompute_freqs_cis(64, compress_ratio + 1, 0, 10000, 1, 32, 1).to(
        get_device()
    )
    page_size = 64
    kvcache = torch.zeros(
        1,
        page_size * (head_dim // 2 + 4),
        dtype=torch.uint8,
        device=get_device(),
    )
    # The inactive row deliberately carries an impossible destination. The
    # consumer must return before reading it or deriving seq_len-ratio == -4.
    out_loc = torch.tensor([0, 1 << 28], dtype=torch.int64, device=get_device())

    compress_norm_rope_store(
        kv,
        plan,
        norm_weight=norm_weight,
        norm_eps=1.0e-6,
        freq_cis=freqs_cis,
        out_loc=out_loc,
        kvcache=kvcache,
        page_size=page_size,
        use_fp4=True,
    )
    torch.cuda.synchronize()
