from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet


class StateDomainKind(str, Enum):
    PAGED_TENSOR = "paged_tensor"
    INDIRECTION = "indirection"
    RECURRENT_SLOT = "recurrent_slot"


class ProtectionProperty(str, Enum):
    IDENTITY = "identity"
    PAYLOAD = "payload"
    TRANSFER = "transfer"
    SAFE_REDIRECT = "safe_redirect"
    TOKEN_GATE = "token_gate"


class ConsumerPath(str, Enum):
    GENERIC = "generic"


@dataclass(frozen=True, slots=True, kw_only=True)
class DomainContract:
    name: str
    kind: StateDomainKind
    properties: FrozenSet[ProtectionProperty]


@dataclass(frozen=True, slots=True, kw_only=True)
class ConsumerContract:
    name: str
    domain: str
    path: ConsumerPath
    properties: FrozenSet[ProtectionProperty]


class ProtectionRegistry:
    """Fail-closed registry connecting persistent state to its consumers.

    This is deliberately semantic rather than a feature bitmap. Kernel families
    declare the concrete properties they preserve, and startup reports the
    missing property by name.
    """

    def __init__(self) -> None:
        self._domains: dict[str, DomainContract] = {}
        self._consumers: list[ConsumerContract] = []

    @property
    def domains(self) -> tuple[DomainContract, ...]:
        return tuple(self._domains.values())

    @property
    def consumers(self) -> tuple[ConsumerContract, ...]:
        return tuple(self._consumers)

    def add_domain(self, contract: DomainContract) -> None:
        previous = self._domains.setdefault(contract.name, contract)
        if previous != contract:
            raise RuntimeError(
                f"state-protection domain {contract.name!r} was registered twice "
                "with different contracts"
            )

    def add_consumer(self, contract: ConsumerContract) -> None:
        if contract.domain not in self._domains:
            raise RuntimeError(
                f"state-protection consumer {contract.name!r} references unknown "
                f"domain {contract.domain!r}"
            )
        self._consumers.append(contract)

    def assert_complete(self) -> None:
        by_domain: dict[str, list[ConsumerContract]] = {
            name: [] for name in self._domains
        }
        for consumer in self._consumers:
            by_domain[consumer.domain].append(consumer)

        errors: list[str] = []
        for name, domain in self._domains.items():
            consumers = by_domain[name]
            if not consumers:
                errors.append(f"{name}: no registered consumer")
                continue
            covered = frozenset(
                prop for consumer in consumers for prop in consumer.properties
            )
            missing = domain.properties - covered
            if missing:
                errors.append(
                    f"{name}: missing {sorted(prop.value for prop in missing)}"
                )
        if errors:
            raise RuntimeError(
                "state protection cannot cover the selected model/backend:\n  - "
                + "\n  - ".join(errors)
            )
