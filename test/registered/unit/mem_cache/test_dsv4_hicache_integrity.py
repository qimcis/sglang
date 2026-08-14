from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.dsv4_kv_integrity import (
    DSV4Component,
    DSV4ComponentDescriptor,
    DSV4IntegrityDomain,
    DSV4TransferGroup,
)
from sglang.srt.mem_cache.memory_pool_host import _DSV4HostIntegritySidecars
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _host_sidecars():
    descriptor = DSV4ComponentDescriptor(
        component=DSV4Component.C4_ATTENTION_KV,
        layer_id=7,
        compress_ratio=4,
        transfer_group=DSV4TransferGroup.KV,
        page_size=64,
        item_nbytes=16,
        capacity=4,
        buffer=torch.zeros((4, 16), dtype=torch.uint8),
    )
    component_sidecar = SimpleNamespace(
        digest=torch.tensor([0, 11, 22, 33], dtype=torch.int64),
        valid=torch.tensor([0, 1, 1, 1], dtype=torch.uint8),
        refresh=lambda pages: None,
    )
    address_space = SimpleNamespace(
        generation=torch.tensor([0, 4, 5, 6], dtype=torch.int64)
    )
    manager = SimpleNamespace(
        sidecars={descriptor.identity: component_sidecar},
        address_spaces={DSV4IntegrityDomain.FULL: address_space},
        domain_for_group=lambda group: DSV4IntegrityDomain.FULL,
    )
    return _DSV4HostIntegritySidecars("c4", manager, [descriptor], 4)


def test_hicache_metadata_detects_corruption_and_reuse():
    sidecars = _host_sidecars()
    host_pages = torch.tensor([1], dtype=torch.int64)
    sidecars.backup(host_pages, torch.tensor([2], dtype=torch.int64))
    assert sidecars.expected_for_restore(0, host_pages).tolist() == [22]

    sidecars.root[1, 0] ^= 1
    with pytest.raises(RuntimeError, match="metadata checksum"):
        sidecars.expected_for_restore(0, host_pages)

    sidecars = _host_sidecars()
    sidecars.backup(host_pages, torch.tensor([2], dtype=torch.int64))
    sidecars.invalidate(host_pages)
    with pytest.raises(RuntimeError, match="metadata is missing"):
        sidecars.expected_for_restore(0, host_pages)


def test_hicache_metadata_rejects_out_of_range_host_pages():
    sidecars = _host_sidecars()
    with pytest.raises(RuntimeError, match="out of range"):
        sidecars.invalidate(torch.tensor([-1], dtype=torch.int64))
