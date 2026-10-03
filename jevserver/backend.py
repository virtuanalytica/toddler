"""Probability backend: a local llama.cpp server, probabilities read from token probabilities.

For a question we ask the model to answer with exactly one option label and read the
probability mass the model puts on each label's first token (llama.cpp /completion with
n_probs). Probabilities are renormalised over the offered labels. This measures the model's
belief; calibration against real labelled data is a separate, measured layer (calibrate.py).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Protocol, Sequence

import requests

LLAMA_URL = os.environ.get("TODDLER_JEV_LLAMA_URL", "http://127.0.0.1:8011")  # the shared --fit llama-server


class Backend(Protocol):
    def option_probs(self, state: object, instructions: str, options: Sequence[str]) -> tuple[dict[str, float], int, int]:
        """Return ({option: probability}, input_tokens, output_tokens)."""


def build_prompt(state: object, instructions: str, options: Sequence[str]) -> str:
    labels = ", ".join(options)
    return (
        "<|im_start|>system\nYou judge situations. Answer with exactly one word from the allowed answers."
        "<|im_end|>\n<|im_start|>user\n"
        f"Situation: {state}\nQuestion: {instructions}\nAllowed answers: {labels}\n"
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _candidates(top: dict) -> list[tuple[str, float]]:
    """Token candidates with probabilities, for both llama.cpp response formats:
    newer `top_logprobs` (logprob) and older `top_probs`/`probs` (prob)."""
    if top.get("top_logprobs"):
        return [(c.get("token", ""), math.exp(c["logprob"])) for c in top["top_logprobs"]]
    return [(c.get("token") or c.get("tok_str") or "", float(c.get("prob", c.get("probability", 0.0))))
            for c in (top.get("top_probs") or top.get("probs") or [])]


@dataclass
class LlamaCppBackend:
    url: str = LLAMA_URL
    timeout_s: float = 30.0
    n_probs: int = 20

    def option_probs(self, state, instructions, options):
        prompt = build_prompt(state, instructions, options)
        r = requests.post(f"{self.url}/completion", json={
            "prompt": prompt, "n_predict": 1, "n_probs": self.n_probs, "temperature": 0, "cache_prompt": True,
        }, timeout=self.timeout_s)
        r.raise_for_status()
        body = r.json()
        top = body["completion_probabilities"][0]
        mass = {o: 0.0 for o in options}
        lower = {o: o.lower() for o in options}
        for tok, p in _candidates(top):
            t = tok.strip().lower()
            if not t:
                continue
            exact = [o for o, ol in lower.items() if ol == t]
            if exact:                         # an exact label always wins
                mass[exact[0]] += p
                continue
            prefixed = [o for o, ol in lower.items() if len(t) >= 2 and ol.startswith(t)]
            if len(prefixed) == 1:            # a prefix counts only when it is unambiguous
                mass[prefixed[0]] += p
        total = sum(mass.values())
        if total <= 0:
            raise RuntimeError("model put no probability on any allowed answer")
        tokens = body.get("tokens_evaluated", 0), body.get("tokens_predicted", 1)
        return {o: m / total for o, m in mass.items()}, int(tokens[0]), int(tokens[1])
