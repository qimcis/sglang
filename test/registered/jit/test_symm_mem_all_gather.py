"""Correctness test for the symmetric-memory multimem all-gather kernel.

Compares ``all_gather_inner`` (concat-along-hidden multimem.st gather) against
NCCL all-gather for a sweep of token counts, hidden widths, and the
``safe`` / ``skip_entry_sync`` knobs, in both eager and CUDA-graph modes.

Usage::

    # Run on the default world sizes (2, 4, 8 GPUs):
    python test/registered/jit/test_symm_mem_all_gather.py
    # Pick a specific world size (or comma-separated list):
    python test/registered/jit/test_symm_mem_all_gather.py --num-gpu 4
    python test/registered/jit/test_symm_mem_all_gather.py --num-gpu 2,4,8
    # Extra pytest args (forwarded to each torchrun worker):
    python test/registered/jit/test_symm_mem_all_gather.py -k 16384
"""

from __future__ import annotations

import atexit
import logging
import os

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.jit_kernel.tests.utils import multigpu_pytest_main
from sglang.jit_kernel.utils import cache_once, get_ci_test_range
from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
    all_gather_inner,
    create_state,
)
from sglang.srt.layers.dp_attention import memcpy_triton
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="base-b-kernel-unit", runner_config="8-gpu-h200")
register_cuda_ci(est_time=240, suite="nightly-kernel-8-gpu-h200", nightly=True)

# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

# Full gathered hidden width H (per-rank shard is H / world_size). Each value
# is a multiple of 8 * 8 so it stays valid for world sizes 2 / 4 / 8.
TEST_HIDDEN = [2048, 7168, 16384]
TEST_NUM_TOKENS = [1, 8, 16, 128]
TEST_LOOP = 8

TEST_HIDDEN = get_ci_test_range(TEST_HIDDEN, [7168])
TEST_NUM_TOKENS = get_ci_test_range(TEST_NUM_TOKENS, [16])

MAX_HIDDEN = max(TEST_HIDDEN)
MAX_TOKENS = max(TEST_NUM_TOKENS)

# ---------------------------------------------------------------------------
# Per-rank distributed setup (run once per torchrun worker)
# ---------------------------------------------------------------------------


@cache_once
def _init_cpu_group_once() -> dist.ProcessGroup:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)
    torch.cuda.set_stream(torch.cuda.Stream())
    cpu_group = ps._WORLD.cpu_group
    assert isinstance(cpu_group, dist.ProcessGroup)
    return cpu_group


@cache_once
def _init_nccl_group_once() -> dist.ProcessGroup:
    _init_cpu_group_once()
    coord = ps._WORLD
    assert coord is not None and coord.device_group is not None
    return coord.device_group


@cache_once
def _init_state_once():
    _init_cpu_group_once()
    coord = ps._WORLD
    return create_state(
        group=coord.device_group,
        rank_in_group=coord.rank_in_group,
        max_tokens=MAX_TOKENS,
        hidden_size=MAX_HIDDEN,
        enable_failure_sideband=True,
    )


def _nccl_all_gather(x: torch.Tensor, group: dist.ProcessGroup, world_size: int):
    """Reference gather matching ``tensor_model_parallel_all_gather(dim=-1)``:
    concat per-rank ``[T, H/W]`` shards in rank order into ``[T, H]``."""
    num_tokens, local_hidden = x.shape
    gathered = torch.empty(
        world_size * num_tokens, local_hidden, dtype=x.dtype, device=x.device
    )
    dist.all_gather_into_tensor(gathered, x.contiguous(), group=group)
    return (
        gathered.reshape(world_size, num_tokens, local_hidden)
        .movedim(0, 1)
        .reshape(num_tokens, world_size * local_hidden)
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@torch.inference_mode()
def test_dp_logits_scatter_copies_failure_sideband() -> None:
    _init_cpu_group_once()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    total_rows, local_capacity, hidden = 11, 6, 32
    local_start, local_rows = 3, 5

    global_logits = torch.arange(
        total_rows * hidden, dtype=torch.float32, device=device
    ).reshape(total_rows, hidden)
    local_logits = torch.full(
        (local_capacity, hidden), -1, dtype=torch.float32, device=device
    )
    global_failure = torch.arange(total_rows, dtype=torch.int32, device=device) % 2
    local_failure = torch.full((local_capacity,), -1, dtype=torch.int32, device=device)

    memcpy_triton(
        local_logits,
        global_logits,
        0,
        torch.tensor(local_start, dtype=torch.int64, device=device),
        torch.tensor(local_rows, dtype=torch.int64, device=device),
        True,
        local_sideband=local_failure,
        global_sideband=global_failure,
    )

    torch.testing.assert_close(
        local_logits[:local_rows],
        global_logits[local_start : local_start + local_rows],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        local_failure[:local_rows],
        global_failure[local_start : local_start + local_rows],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        local_logits[local_rows:],
        torch.full_like(local_logits[local_rows:], -1),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        local_failure[local_rows:],
        torch.full_like(local_failure[local_rows:], -1),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("skip_entry_sync", [False, True])
@pytest.mark.parametrize("safe", [False, True])
@pytest.mark.parametrize("hidden", TEST_HIDDEN)
@pytest.mark.parametrize("num_tokens", TEST_NUM_TOKENS)
@torch.inference_mode()
def test_symm_mem_all_gather(
    num_tokens: int,
    hidden: int,
    safe: bool,
    skip_entry_sync: bool,
) -> None:
    nccl_group = _init_nccl_group_once()
    state = _init_state_once()
    world_size = state.world_size
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")

    if state.symm_mem_hdl.multicast_ptr == 0:
        pytest.skip(f"multimem multicast unavailable for world_size={world_size}")

    local_hidden = hidden // world_size
    if hidden % world_size != 0 or local_hidden % 8 != 0:
        pytest.skip(f"hidden={hidden} incompatible with world_size={world_size}")

    def gather(
        x: torch.Tensor,
        local_failure: torch.Tensor | None = None,
        global_failure: torch.Tensor | None = None,
        failure_local_start: torch.Tensor | None = None,
        failure_local_num_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return all_gather_inner(
            state,
            x,
            tp_hidden_dim=hidden,
            skip_entry_sync=skip_entry_sync,
            safe=safe,
            local_failure=local_failure,
            global_failure=global_failure,
            failure_local_start=failure_local_start,
            failure_local_num_rows=failure_local_num_rows,
        ).clone()

    for _ in range(TEST_LOOP):
        # Entry barrier may be skipped on the kernel side; make sure every rank's
        # input is ready and the buffer is free before the next gather.
        dist.barrier(nccl_group)
        x = torch.randn(num_tokens, local_hidden, dtype=torch.bfloat16, device=device)
        ref = _nccl_all_gather(x, nccl_group, world_size)
        out = gather(x)
        # Pure copy gather: exact bitwise equality.
        torch.testing.assert_close(out, ref, atol=0, rtol=0)

        if state.failure_symm_mem_hdl.multicast_ptr == 0:
            continue
        local_failure = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        local_failure[state.rank_in_group % num_tokens] = 1
        global_failure = torch.full_like(local_failure, -1)
        out = gather(x, local_failure, global_failure)
        expected_failure = torch.zeros_like(local_failure)
        expected_failure[: min(world_size, num_tokens)] = 1
        torch.testing.assert_close(out, ref, atol=0, rtol=0)
        torch.testing.assert_close(global_failure, expected_failure, atol=0, rtol=0)

        # DPA lanes own disjoint row ranges. Every attention-TP rank publishes
        # only its lane-local vector; the full TP gather must OR matching ranks
        # without confusing equal local batch indices from different lanes.
        if num_tokens >= 8 and world_size % 2 == 0:
            dp_size = 2
            attn_tp_size = world_size // dp_size
            dp_rank = state.rank_in_group // attn_tp_size
            attn_tp_rank = state.rank_in_group % attn_tp_size
            local_num_rows = num_tokens // dp_size
            local_start = dp_rank * local_num_rows
            local_failure = torch.zeros(
                local_num_rows, dtype=torch.int32, device=device
            )
            local_failure[attn_tp_rank % local_num_rows] = 1
            global_failure = torch.full(
                (num_tokens,), -1, dtype=torch.int32, device=device
            )
            out = gather(
                x,
                local_failure,
                global_failure,
                torch.tensor(local_start, dtype=torch.int32, device=device),
                torch.tensor(local_num_rows, dtype=torch.int32, device=device),
            )
            expected_failure = torch.zeros_like(global_failure)
            for lane in range(dp_size):
                lane_start = lane * local_num_rows
                expected_failure[
                    lane_start : lane_start + min(attn_tp_size, local_num_rows)
                ] = 1
            torch.testing.assert_close(out, ref, atol=0, rtol=0)
            torch.testing.assert_close(global_failure, expected_failure, atol=0, rtol=0)

            # Uneven lanes use cumulative global offsets rather than fixed-size
            # slots. Verify the sideband follows the same compact row layout.
            uneven_counts = (3, num_tokens - 3)
            uneven_local_rows = uneven_counts[dp_rank]
            uneven_local_start = sum(uneven_counts[:dp_rank])
            uneven_failure = torch.zeros(
                uneven_local_rows, dtype=torch.int32, device=device
            )
            uneven_failure[attn_tp_rank % uneven_local_rows] = 1
            global_failure.fill_(-1)
            out = gather(
                x,
                uneven_failure,
                global_failure,
                torch.tensor(uneven_local_start, dtype=torch.int32, device=device),
                torch.tensor(uneven_local_rows, dtype=torch.int32, device=device),
            )
            expected_failure.zero_()
            for lane, lane_rows in enumerate(uneven_counts):
                lane_start = sum(uneven_counts[:lane])
                expected_failure[
                    lane_start : lane_start + min(attn_tp_size, lane_rows)
                ] = 1
            torch.testing.assert_close(out, ref, atol=0, rtol=0)
            torch.testing.assert_close(global_failure, expected_failure, atol=0, rtol=0)

            # An idle lane contributes no failures to another lane's rows.
            if dp_rank == 0:
                idle_local_start = 0
                idle_local_rows = num_tokens
                idle_failure = torch.zeros(num_tokens, dtype=torch.int32, device=device)
                idle_failure[attn_tp_rank % num_tokens] = 1
            else:
                idle_local_start = num_tokens
                idle_local_rows = 0
                idle_failure = torch.zeros(1, dtype=torch.int32, device=device)
            global_failure.fill_(-1)
            out = gather(
                x,
                idle_failure,
                global_failure,
                torch.tensor(idle_local_start, dtype=torch.int32, device=device),
                torch.tensor(idle_local_rows, dtype=torch.int32, device=device),
            )
            expected_failure.zero_()
            expected_failure[: min(attn_tp_size, num_tokens)] = 1
            torch.testing.assert_close(out, ref, atol=0, rtol=0)
            torch.testing.assert_close(global_failure, expected_failure, atol=0, rtol=0)


if __name__ == "__main__":
    # multimem multicast needs world_size in {4, 6, 8} (cc9) or {6, 8} (cc10);
    # unsupported sizes self-skip via the multicast_ptr guard above.
    multigpu_pytest_main(__name__, __file__, num_gpus=(4, 8))
