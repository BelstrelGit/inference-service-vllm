"""Synthetic prompts with an exact number of input tokens (EXPERIMENT_PROTOCOL.md, section 4).

Two formats:
  token_ids  the prompt is sent as a list of token ids (/v1/completions accepts that), so the
             server sees exactly `input_len` tokens and tokenization drops out of the path.
             Default for controlled experiments.
  text       random ids are turned into text via the server's own tokenizer (/detokenize),
             then re-encoded (/tokenize) and trimmed/extended until the text encodes to
             exactly `input_len` tokens. Closer to real traffic and to `vllm bench serve`
             (its random dataset does the same decode/encode loop).

Every prompt is random, so prompts share no prefix and prefix caching gets no hits, unless a
shared prefix is asked for explicitly (`shared_prefix_len`, for prefix-caching experiments).
Random ids are drawn from [token_id_low, token_id_high): the default range stays inside the
regular vocabulary of Qwen2.5 (special tokens start at 151643).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import httpx


@dataclass
class PromptSpec:
    input_len: int
    format: str = "token_ids"  # "token_ids" | "text"
    shared_prefix_len: int = 0
    token_id_low: int = 1000
    token_id_high: int = 150000
    seed: int = 0


class PromptFactory:
    def __init__(self, spec: PromptSpec, client: httpx.AsyncClient | None = None, model: str = ""):
        if spec.shared_prefix_len > spec.input_len:
            raise ValueError("shared_prefix_len > input_len")
        if spec.format not in ("token_ids", "text"):
            raise ValueError(f"unknown prompt format: {spec.format}")
        if spec.format == "text" and client is None:
            raise ValueError("text prompts need a client for /tokenize and /detokenize")
        self.spec = spec
        self.rng = random.Random(spec.seed)
        self.client = client
        self.model = model
        self.prefix = self._ids(spec.shared_prefix_len)

    def _ids(self, n: int) -> list[int]:
        return [
            self.rng.randrange(self.spec.token_id_low, self.spec.token_id_high) for _ in range(n)
        ]

    async def next(self) -> str | list[int]:
        ids = self.prefix + self._ids(self.spec.input_len - len(self.prefix))
        if self.spec.format == "token_ids":
            return ids
        return await self._fit_text(ids)

    async def _detokenize(self, ids: list[int]) -> str:
        r = await self.client.post("/detokenize", json={"model": self.model, "tokens": ids})
        r.raise_for_status()
        return r.json()["prompt"]

    async def _tokenize(self, text: str) -> list[int]:
        r = await self.client.post(
            "/tokenize", json={"model": self.model, "prompt": text, "add_special_tokens": False}
        )
        r.raise_for_status()
        return r.json()["tokens"]

    async def _fit_text(self, ids: list[int], max_rounds: int = 10) -> str:
        """decode -> encode can merge or split tokens; adjust until the length is exact."""
        target = self.spec.input_len
        for _ in range(max_rounds):
            text = await self._detokenize(ids)
            enc = await self._tokenize(text)
            if len(enc) == target:
                return text
            ids = enc[:target] if len(enc) > target else enc + self._ids(target - len(enc))
        raise RuntimeError(f"could not build a text prompt of exactly {target} tokens")
