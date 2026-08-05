from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.state_protection.contracts import (
    ConsumerContract,
    ConsumerPath,
    DomainContract,
    ProtectionProperty,
    ProtectionRegistry,
    StateDomainKind,
)
from sglang.srt.state_protection.paged import PagedStateProtection
from sglang.srt.state_protection.recurrent import RecurrentStateProtection

if TYPE_CHECKING:
    from sglang.srt.kv_canary.runner.canary_manager import CanaryManager
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

STATE_PROTECTION_INVALID_REQUEST = 1 << 23
STATE_PROTECTION_REQUEST_IDENTITY = 1 << 22
STATE_PROTECTION_CANARY_FAILURE = 1 << 24
STATE_PROTECTION_REMOTE_FAILURE = 1 << 30


@dataclass(frozen=True, slots=True, kw_only=True)
class StateProtectionCheck:
    request_indices: torch.Tensor
    statuses: torch.Tensor
    failed: torch.Tensor

    def copy_to_cpu(self) -> None:
        from sglang.srt.managers.utils import _async_d2h

        object.__setattr__(self, "request_indices", _async_d2h(self.request_indices))
        object.__setattr__(self, "statuses", _async_d2h(self.statuses))
        object.__setattr__(self, "failed", _async_d2h(self.failed))

    def mask_failed_rows(
        self, values: torch.Tensor, fallback: int | torch.Tensor = 0
    ) -> torch.Tensor:
        failed = self.failed.to(device=values.device, dtype=torch.bool)
        while failed.ndim < values.ndim:
            failed = failed.unsqueeze(-1)
        fallback_tensor = torch.as_tensor(
            fallback, dtype=values.dtype, device=values.device
        )
        return torch.where(failed, fallback_tensor, values)

    def materialize(self) -> tuple[list[int], list[int], list[bool]]:
        return (
            self.request_indices.tolist(),
            self.statuses.tolist(),
            self.failed.bool().tolist(),
        )


class StateProtectionManager:
    def __init__(
        self,
        *,
        model_runner: ModelRunner,
        canary_manager: CanaryManager,
    ) -> None:
        self.model_runner = model_runner
        self.canary_manager = canary_manager
        req_slots = int(model_runner.req_to_token_pool.req_to_token.shape[0])
        self.enabled = torch.zeros(1, dtype=torch.int32, device=model_runner.device)
        self._initialized = False
        self.failure_status = torch.zeros(
            req_slots, dtype=torch.int32, device=model_runner.device
        )
        self._canary_start = torch.zeros(
            1, dtype=torch.int32, device=model_runner.device
        )
        self._expected_requests: Optional[torch.Tensor] = None
        self.registry = ProtectionRegistry()

        paged = PagedStateProtection(
            model_runner=model_runner,
            failure_status=self.failure_status,
            enabled=self.enabled,
            canary_violation_index=self.canary_violation_index,
            canary_forward_start=self._canary_start,
        )
        self.paged: Optional[PagedStateProtection] = paged if paged.layers else None
        if self.paged is not None:
            properties = frozenset(
                {
                    ProtectionProperty.IDENTITY,
                    ProtectionProperty.PAYLOAD,
                    ProtectionProperty.TRANSFER,
                    ProtectionProperty.SAFE_REDIRECT,
                    ProtectionProperty.TOKEN_GATE,
                }
            )
            self.registry.add_domain(
                DomainContract(
                    name="paged-cache",
                    kind=StateDomainKind.PAGED_TENSOR,
                    properties=properties,
                )
            )
            self.registry.add_consumer(
                ConsumerContract(
                    name="paged-attention-accessor",
                    domain="paged-cache",
                    path=ConsumerPath.GENERIC,
                    properties=properties,
                )
            )

        req_pool = model_runner.req_to_token_pool
        if hasattr(req_pool, "mamba_pool"):
            recurrent_properties = frozenset(
                {
                    ProtectionProperty.IDENTITY,
                    ProtectionProperty.PAYLOAD,
                    ProtectionProperty.TRANSFER,
                    ProtectionProperty.SAFE_REDIRECT,
                    ProtectionProperty.TOKEN_GATE,
                }
            )
            self.recurrent: Optional[RecurrentStateProtection] = (
                RecurrentStateProtection(
                    req_to_token_pool=req_pool,
                    failure_status=self.failure_status,
                    enabled=self.enabled,
                    canary_violation_index=self.canary_violation_index,
                    canary_forward_start=self._canary_start,
                )
            )
            self.registry.add_domain(
                DomainContract(
                    name="recurrent-state",
                    kind=StateDomainKind.RECURRENT_SLOT,
                    properties=recurrent_properties,
                )
            )
            self.registry.add_consumer(
                ConsumerContract(
                    name="recurrent-state-accessor",
                    domain="recurrent-state",
                    path=ConsumerPath.GENERIC,
                    properties=recurrent_properties,
                )
            )
        else:
            self.recurrent = None

        if self.paged is None and self.recurrent is None:
            raise RuntimeError(
                "state protection found neither a supported paged cache nor a "
                "supported recurrent state pool"
            )
        self.registry.assert_complete()

    @property
    def canary_violation_index(self) -> torch.Tensor:
        return self.canary_manager.violation_write_index

    def mark_init_finished(self) -> None:
        self.enabled.fill_(1)
        self._initialized = True

    def bind_backend(self, backend) -> None:
        """Attach the manager recursively to composite attention backends."""

        seen: set[int] = set()

        def visit(node) -> None:
            if node is None or id(node) in seen:
                return
            seen.add(id(node))
            if (
                self.recurrent is not None
                and type(node).__name__ == "InklingShortConvAttnBackend"
            ):
                raise RuntimeError(
                    "--enable-state-protection does not yet cover Inkling's "
                    "fused all-reduce/short-convolution consumers"
                )
            node.state_protection_manager = self
            linear_backend = getattr(node, "linear_attn_backend", None)
            if (
                self.recurrent is not None
                and linear_backend is not None
                and not callable(
                    getattr(linear_backend, "recurrent_state_guard", None)
                )
            ):
                raise RuntimeError(
                    "state protection found a recurrent attention backend "
                    f"without an accessor integration point: "
                    f"{type(linear_backend).__name__}"
                )
            for attr in (
                "attn_backends",
                "attn_backend_list",
                "children",
                "primary",
                "decode_backend",
                "prefill_backend",
                "full_attn_backend",
                "linear_attn_backend",
                "short_conv_backend",
                "dense",
                "sparse",
            ):
                child = getattr(node, attr, None)
                if isinstance(child, str):
                    continue
                if isinstance(child, (tuple, list)):
                    for item in child:
                        visit(item)
                else:
                    visit(child)

        visit(backend)

    def begin_forward(self, forward_batch) -> None:
        requests = forward_batch.req_pool_indices[: forward_batch.batch_size]
        if requests.dtype != torch.int64 or not requests.is_contiguous():
            raise RuntimeError(
                "state protection requires contiguous int64 request-pool indices"
            )
        # Keep an independent per-forward identity snapshot before any model or
        # canary kernel can consume/mutate the live routing tensor. This makes a
        # valid-in-range request-index corruption visible before sampling and TP
        # consensus, rather than only when the scheduler receives the result.
        self._expected_requests = requests.clone()
        self.failure_status.zero_()
        invalid_requests = requests.lt(0) | requests.ge(self.failure_status.shape[0])
        self.failure_status[0].bitwise_or_(
            invalid_requests.any().to(torch.int32)
            * STATE_PROTECTION_INVALID_REQUEST
        )
        # Request slot zero is the reserved sink used by every protected
        # accessor. Sanitizing here happens before attention/recurrent metadata
        # consumes the indices and is captured into decode CUDA graphs.
        requests.masked_fill_(invalid_requests, 0)
        self._canary_start.copy_(self.canary_violation_index)
        if self.paged is not None and not forward_batch.forward_mode.is_idle():
            self.paged.preflight_mapping(forward_batch)

    def finish_forward(self, forward_batch) -> Optional[StateProtectionCheck]:
        if not self._initialized:
            return None
        requests = forward_batch.req_pool_indices[: forward_batch.batch_size]
        expected_requests = self._expected_requests
        if expected_requests is None or expected_requests.shape != requests.shape:
            raise RuntimeError("state-protection request identity snapshot mismatch")
        invalid_requests = requests.lt(0) | requests.ge(self.failure_status.shape[0])
        self.failure_status[0].bitwise_or_(
            invalid_requests.any().to(torch.int32)
            * STATE_PROTECTION_INVALID_REQUEST
        )
        requests.masked_fill_(invalid_requests, 0)
        if requests.numel() == 0:
            empty_status = self.failure_status[:0].clone()
            return StateProtectionCheck(
                request_indices=requests.clone(),
                statuses=empty_status,
                failed=empty_status.to(torch.bool),
            )
        statuses = self.failure_status.index_select(0, requests).clone()
        statuses.bitwise_or_(self.failure_status[:1])
        canary_failed = self.canary_violation_index.ne(self._canary_start)
        statuses.bitwise_or_(
            canary_failed.to(torch.int32) * STATE_PROTECTION_CANARY_FAILURE
        )
        identity_failed = requests.ne(expected_requests)
        statuses.bitwise_or_(
            identity_failed.to(torch.int32)
            * STATE_PROTECTION_REQUEST_IDENTITY
        )

        failed = statuses.ne(0).to(torch.int32)
        from sglang.srt.distributed.parallel_state import get_attn_tp_group

        attn_tp_group = get_attn_tp_group()
        if attn_tp_group.world_size > 1:
            failed = attn_tp_group.all_reduce(failed)
        failed = failed.ne(0)
        remote_only = failed & statuses.eq(0)
        statuses.bitwise_or_(
            remote_only.to(torch.int32) * STATE_PROTECTION_REMOTE_FAILURE
        )
        return StateProtectionCheck(
            request_indices=requests.clone(),
            statuses=statuses,
            failed=failed,
        )


def install_state_protection(
    *,
    server_args: ServerArgs,
    model_runner: ModelRunner,
    canary_manager: Optional[CanaryManager],
) -> Optional[StateProtectionManager]:
    if not server_args.enable_state_protection:
        return None
    if torch.device(model_runner.device).type != "cuda":
        raise RuntimeError("--enable-state-protection currently requires CUDA")
    if not model_runner.is_generation:
        raise RuntimeError(
            "--enable-state-protection currently supports generation models only"
        )
    if model_runner.model_config.is_encoder_decoder:
        raise RuntimeError(
            "--enable-state-protection does not yet cover encoder/cross-attention state"
        )
    if canary_manager is None:
        raise RuntimeError(
            "state protection requires the paged-state canary to be installed"
        )
    manager = StateProtectionManager(
        model_runner=model_runner,
        canary_manager=canary_manager,
    )
    logger.info(
        "state protection enabled: domains=%s consumers=%s",
        [domain.name for domain in manager.registry.domains],
        [consumer.name for consumer in manager.registry.consumers],
    )
    return manager
