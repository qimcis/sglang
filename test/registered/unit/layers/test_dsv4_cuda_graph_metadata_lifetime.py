import gc
import weakref
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.deepseek_v4_backend import (
    DeepseekV4AttnBackend,
    DSV4Metadata,
    DSV4RawDecodeMetadata,
    _GraphBucket,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _raw_decode_metadata(batch_size: int) -> DSV4RawDecodeMetadata:
    return DSV4RawDecodeMetadata(
        req_pool_indices=torch.zeros(batch_size, dtype=torch.int32),
        seq_lens=torch.ones(batch_size, dtype=torch.int32),
        out_cache_loc=torch.zeros(batch_size, dtype=torch.int32),
    )


def _captured_metadata() -> DSV4Metadata:
    return DSV4Metadata(core_attn_metadata=SimpleNamespace(), indexer_metadata=None)


def test_full_capture_metadata_is_retained_for_every_graph_shape() -> None:
    backend = object.__new__(DeepseekV4AttnBackend)
    backend.cuda_graph_captured_metadata_of_bucket_and_bs = {
        bucket: {} for bucket in _GraphBucket
    }
    backend._integrity_protect_forward_metadata = lambda metadata: None
    forward_batch = SimpleNamespace(out_cache_loc=None)
    bucket = _GraphBucket.DECODE_OR_IDLE

    first = _captured_metadata()
    first_ref = weakref.ref(first)
    backend.forward_metadata = _raw_decode_metadata(2)
    backend._current_capture_metadata_key = (bucket, 2)
    backend.make_forward_metadata_from_raw_decode = (
        lambda raw_metadata, captured=first: captured
    )
    backend.init_forward_metadata_in_graph(forward_batch)
    assert backend.cuda_graph_captured_metadata_of_bucket_and_bs[bucket][2] == [first]

    second = _captured_metadata()
    second_ref = weakref.ref(second)
    backend.forward_metadata = _raw_decode_metadata(4)
    backend._current_capture_metadata_key = (bucket, 4)
    backend.make_forward_metadata_from_raw_decode = (
        lambda raw_metadata, captured=second: captured
    )
    backend.init_forward_metadata_in_graph(forward_batch)
    assert backend.cuda_graph_captured_metadata_of_bucket_and_bs[bucket][4] == [second]

    backend.forward_metadata = None
    backend.make_forward_metadata_from_raw_decode = lambda raw_metadata: None
    del first
    del second
    gc.collect()

    assert first_ref() is not None
    assert second_ref() is not None


def test_warmups_are_dropped_but_same_shape_graph_variants_are_retained() -> None:
    backend = object.__new__(DeepseekV4AttnBackend)
    backend.cuda_graph_captured_metadata_of_bucket_and_bs = {
        bucket: {} for bucket in _GraphBucket
    }
    backend._integrity_protect_forward_metadata = lambda metadata: None
    forward_batch = SimpleNamespace(out_cache_loc=None)
    bucket = _GraphBucket.DECODE_OR_IDLE
    backend._current_capture_metadata_key = (bucket, 2)

    def run(full: DSV4Metadata, *, warmup: bool) -> None:
        backend.forward_metadata = _raw_decode_metadata(2)
        backend._current_capture_raw = backend.forward_metadata
        backend.make_forward_metadata_from_raw_decode = lambda raw_metadata: full
        backend.init_forward_metadata_in_graph(forward_batch)
        if warmup:
            backend.on_after_cuda_graph_warmup()

    warmup = _captured_metadata()
    warmup_ref = weakref.ref(warmup)
    run(warmup, warmup=True)
    assert 2 not in backend.cuda_graph_captured_metadata_of_bucket_and_bs[bucket]
    backend.make_forward_metadata_from_raw_decode = lambda raw_metadata: None
    del warmup
    gc.collect()
    assert warmup_ref() is None

    first_variant = _captured_metadata()
    second_variant = _captured_metadata()
    run(first_variant, warmup=False)
    run(second_variant, warmup=False)

    assert backend.cuda_graph_captured_metadata_of_bucket_and_bs[bucket][2] == [
        first_variant,
        second_variant,
    ]
