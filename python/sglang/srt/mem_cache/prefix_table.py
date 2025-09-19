from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class PrefixEntry:
    device_indices: torch.Tensor
    last_node: object
    refcount: int = 0


class PrefixTable:
    """Simple hashed prefix overlay mapping fixed-length token prefixes to KV handles.

    This table holds references to radix cache nodes and their device indices
    for quick reuse. It is advisory: eviction still governed by the radix cache.
    """

    def __init__(self, hash_len: int):
        self.hash_len = hash_len
        self._table: Dict[str, PrefixEntry] = {}

    @staticmethod
    def _hash_tokens(tokens: torch.Tensor) -> str:
        # tokens: 1-D tensor/list of ints
        if isinstance(tokens, torch.Tensor):
            data = tokens.to(torch.int32, copy=False).cpu().numpy().tobytes()
        else:
            data = bytes(int(x) & 0xFFFFFFFF for x in tokens)
        return hashlib.sha1(data).hexdigest()

    def lookup(self, tokens: torch.Tensor) -> Optional[PrefixEntry]:
        if tokens is None:
            return None
        if isinstance(tokens, list):
            if len(tokens) < self.hash_len:
                return None
            key = self._hash_tokens(torch.tensor(tokens[: self.hash_len]))
        else:
            if tokens.numel() < self.hash_len:
                return None
            key = self._hash_tokens(tokens[: self.hash_len])
        return self._table.get(key)

    def attach(self, tokens: torch.Tensor) -> Optional[PrefixEntry]:
        entry = self.lookup(tokens)
        if entry is not None:
            entry.refcount += 1
        return entry

    def put(self, tokens: torch.Tensor, device_indices: torch.Tensor, last_node: object):
        if isinstance(tokens, list):
            if len(tokens) < self.hash_len:
                return
            key = self._hash_tokens(torch.tensor(tokens[: self.hash_len]))
        else:
            if tokens.numel() < self.hash_len:
                return
            key = self._hash_tokens(tokens[: self.hash_len])
        if key not in self._table:
            self._table[key] = PrefixEntry(device_indices=device_indices.to(torch.int64, copy=False), last_node=last_node, refcount=0)

