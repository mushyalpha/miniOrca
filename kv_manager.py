from __future__ import annotations

from typing import Optional

import torch


class KVOutOfMemory(RuntimeError):
    def __init__(self, requested: int, available: int):
        super().__init__(f"requested {requested} K/V slots, only {available} free")
        self.requested, self.available = requested, available


class SlotAllocator:

    def __init__(self, n_slots: int):
        self.n_slots = n_slots
        self.used = 0
        self.peak = 0

    @property
    def available(self) -> int:
        return self.n_slots - self.used

    def can_alloc(self, n: int) -> bool:
        return n <= self.available

    def alloc(self, n: int) -> None:
        if n > self.available:
            raise KVOutOfMemory(n, self.available)
        self.used += n
        self.peak = max(self.peak, self.used)

    def release(self, n: int) -> None:
        self.used -= n
        assert self.used >= 0, 

    def bytes_used(self, bytes_per_slot: int) -> int:
        return self.used * bytes_per_slot


class RequestKV:

    def __init__(self, n_layers: int, n_kv_heads: int, head_dim: int,
                 device: torch.device, dtype: torch.dtype,
                 capacity: Optional[int] = None):
        self.n_layers, self.n_kv_heads, self.head_dim = n_layers, n_kv_heads, head_dim
        self.device, self.dtype = device, dtype
        self.capacity = capacity
        self.k: list[Optional[torch.Tensor]] = [None] * n_layers
        self.v: list[Optional[torch.Tensor]] = [None] * n_layers
        self.length = 0          # committed tokens
        self.reserved = 0        # slots charged to the allocator

    def extend(self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor):
        t = k_new.shape[2]
        if self.capacity is not None:
            if self.k[layer] is None:
                shape = (1, self.n_kv_heads, self.capacity, self.head_dim)
                self.k[layer] = torch.empty(shape, device=self.device, dtype=self.dtype)
                self.v[layer] = torch.empty(shape, device=self.device, dtype=self.dtype)
            end = self.length + t
            assert end <= self.capacity, 
            self.k[layer][:, :, self.length:end] = k_new
            self.v[layer][:, :, self.length:end] = v_new
            return self.k[layer][:, :, :end], self.v[layer][:, :, :end]

        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k_new, v_new
        else:
            self.k[layer] = torch.cat([self.k[layer], k_new], dim=2)
            self.v[layer] = torch.cat([self.v[layer], v_new], dim=2)
        return self.k[layer], self.v[layer]

    def commit(self, n_tokens: int) -> None:
        self.length += n_tokens

    def free(self) -> None:
        self.k = [None] * self.n_layers
        self.v = [None] * self.n_layers
