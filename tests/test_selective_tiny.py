from __future__ import annotations

import _path  
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from kv_manager import RequestKV
from model_loader import LoadedModel
from request import RequestState
from selective_batching import selective_batching_forward

torch.manual_seed(0)
VOCAB = 2000


def make_tiny_lm() -> LoadedModel:
    cfg = Qwen2Config(
        vocab_size=VOCAB, hidden_size=32, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, intermediate_size=64,
        max_position_embeddings=1024, rms_norm_eps=1e-6,
        pad_token_id=0, eos_token_id=1,
    )
    model = Qwen2ForCausalLM(cfg)
    model.eval()

    class FakeTok:
        pad_token_id = 0
        vocab_size = VOCAB
        all_special_ids = [0, 1]

    return LoadedModel(
        model=model, tokenizer=FakeTok(), device=torch.device("cpu"), dtype=torch.float32,
        eos_id=1, n_layers=cfg.num_hidden_layers, n_heads=cfg.num_attention_heads,
        n_kv_heads=cfg.num_key_value_heads, head_dim=cfg.hidden_size // cfg.num_attention_heads,
        hidden_size=cfg.hidden_size, vocab_size=cfg.vocab_size,
    )


def validate(lm: LoadedModel) -> None:
    prompts = [[10, 20, 30, 40, 50], list(range(100, 117))]

    ref = []
    for p in prompts:
        out = lm.model(input_ids=torch.tensor([p]))
        ref.append(out.logits[0, -1])

    reqs = [RequestState(rid=i, prompt_ids=p, max_gen_tokens=4, arrival_time=0.0)
            for i, p in enumerate(prompts)]
    kv = {r.rid: RequestKV(lm.n_layers, lm.n_kv_heads, lm.head_dim,
                           lm.device, lm.dtype, capacity=r.max_tokens)
          for r in reqs}
    got = selective_batching_forward(lm, reqs, kv)

    for i in range(len(prompts)):
        d = (got[i] - ref[i]).abs().max().item()
        print(f"  req {i}: max |Δlogit| = {d:.2e}  "
              f"argmax {'OK' if got[i].argmax() == ref[i].argmax() else 'MISMATCH'}")
        assert d < 1e-3, "flat forward diverges from HF"

    for r in reqs:
        r.append_token(int(got[reqs.index(r)].argmax()), 0.0)
    reqs.append(RequestState(rid=99, prompt_ids=[7, 8, 9], max_gen_tokens=4,
                             arrival_time=0.0))
    kv[99] = RequestKV(lm.n_layers, lm.n_kv_heads, lm.head_dim, lm.device,
                       lm.dtype, capacity=reqs[-1].max_tokens)
    out = selective_batching_forward(lm, reqs, kv)
    assert out.shape == (3, lm.vocab_size)
    print("  mixed initiation+increment batch OK", out.shape)


    for r in reqs:
        r.append_token(0, 0.0)
    out2 = selective_batching_forward(lm, reqs, kv)
    assert out2.shape == (3, lm.vocab_size)
    print("  three-way pure-decode batch OK", out2.shape)


if __name__ == "__main__":
    print("=== selective_batching_forward vs HF dense forward (tiny random Qwen2) ===")
    validate(make_tiny_lm())
    print("\nPASS: flatten/RoPE/GQA/split/merge dataflow matches HF exactly.")
