"""One loader for every engine. If dtype or attn impl drifts between
engines, throughput deltas stop meaning anything."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def select_device_and_dtype(force_fp32: bool = False,
                            device: Optional[str] = None
                            ) -> tuple[torch.device, torch.dtype]:
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    dev = torch.device(device)
    if force_fp32:
        return dev, torch.float32
    if dev.type == "cuda":
        return dev, torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if dev.type == "mps":
        return dev, torch.float16
    return dev, torch.float32          # CPU fp16 is slow and badly supported


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: object
    device: torch.device
    dtype: torch.dtype
    eos_id: int
    # cached config, needed by the selective-batching forward
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int

    @property
    def n_rep(self) -> int:
        """GQA repeat factor for repeat_kv."""
        return self.n_heads // self.n_kv_heads

    def bytes_per_slot(self) -> int:
        """One K/V "slot" (§4.2) = K and V for one token, all layers."""
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * self.dtype.itemsize


def load_model_and_tokenizer(model_id: str = "Qwen/Qwen2.5-0.5B",
                             force_fp32: bool = False,
                             device: Optional[str] = None,
                             attn_implementation: str = "sdpa",
                             deterministic: bool = True) -> LoadedModel:
    if deterministic:
        # TF32 changes matmul reduction order; leaving it on makes
        # cross-engine logit comparison flaky for no benefit here.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(0)

    dev, dtype = select_device_and_dtype(force_fp32, device)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation=attn_implementation,
    ).to(dev).eval()

    cfg = model.config
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_heads)
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads
    eos_id = cfg.eos_token_id if isinstance(cfg.eos_token_id, int) else cfg.eos_token_id[0]

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    assert not getattr(cfg, "use_sliding_window", False), \
        "sliding window breaks the per-request full-cache attention assumption"

    return LoadedModel(
        model=model, tokenizer=tok, device=dev, dtype=dtype, eos_id=eos_id,
        n_layers=cfg.num_hidden_layers, n_heads=n_heads, n_kv_heads=n_kv,
        head_dim=head_dim, hidden_size=cfg.hidden_size, vocab_size=cfg.vocab_size,
    )
