"""Jev client for the reflex fast path: TypeSafe System One HTTP API.

Mirrors the documented contract implemented in virtualpc-jev-finance/src/finance/jev-client.ts:
POST {endpoint} with {"state", "questions", "model"}; each answer is typed ('noul' carries a
probability, 'choice'/'score' carry probabilities and a confidence). Every PhysicalQuestion is
sent as a 'noul' question ("is this true?").

Safety: any malformed answer raises, and fastpath.reflex treats a missing or late answer as
STOP. A remote API will rarely meet the 20 ms reflex budget; for physical use the client must
run next to the robot or answer from a cache, otherwise the robot stops, which is the intended
fail-safe.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Sequence

import requests

from toddler.fastpath import Answer, PhysicalQuestion

DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"


class JevResponseError(ValueError):
    pass


def build_request(state: dict, questions: Sequence[PhysicalQuestion], model: str = DEFAULT_MODEL) -> dict:
    return {
        "state": state,
        "model": model,
        "questions": {q.qid: {"type": "noul", "instructions": q.text,
                              "criteria": {"true": "yes", "false": "no"}} for q in questions},
    }


def _prob(x, where: str) -> float:
    if not isinstance(x, (int, float)) or isinstance(x, bool) or not 0.0 <= float(x) <= 1.0:
        raise JevResponseError(f"{where}: not a probability: {x!r}")
    return float(x)


def parse_answers(body: dict, questions: Sequence[PhysicalQuestion], noul_uncertainty: float) -> list[Answer]:
    """Strict parse of the answers Toddler asked for.

    Deliberately lenient in the same two places as the reference schema (jev-client.ts): the
    returned 'model' is not compared with the requested one, and answers for qids Toddler did
    not ask about are ignored. Both are harmless for a reflex, which only reads its own qids.

    Uncertainty: 'noul' carries no confidence, so a configured uncertainty is used (client
    default 0.05: the reflex judges p + uncertainty, and the tightest default slow threshold is
    0.1, so a larger default would keep every calm scene permanently in 'slow'). For
    'choice'/'score' the API's 'confidence' is the probability of the chosen option, so
    1 - confidence is used as the uncertainty; it is not added on top of noul_uncertainty.
    fastpath.reflex then judges p + uncertainty, which can only make Toddler more careful."""
    answers = body.get("answers")
    if not isinstance(answers, dict) or not isinstance(body.get("model"), str):
        raise JevResponseError("response lacks 'model' or 'answers'")
    out: list[Answer] = []
    for q in questions:
        a = answers.get(q.qid)
        if not isinstance(a, dict):
            raise JevResponseError(f"missing answer for {q.qid}")
        kind = a.get("type")
        if kind == "noul":
            out.append(Answer(q.qid, _prob(a.get("noul"), q.qid), noul_uncertainty))
        elif kind in ("choice", "score"):
            probs = a.get("probabilities")
            if not isinstance(probs, dict) or "yes" not in probs:
                raise JevResponseError(f"{q.qid}: no probability for 'yes'")
            out.append(Answer(q.qid, _prob(probs["yes"], q.qid), 1.0 - _prob(a.get("confidence"), q.qid)))
        else:
            raise JevResponseError(f"{q.qid}: unknown answer type {kind!r}")
    return out


class HttpJevClient:
    def __init__(self, api_key: str | None = None, endpoint: str = DEFAULT_ENDPOINT, model: str = DEFAULT_MODEL,
                 timeout_s: float = 0.02, noul_uncertainty: float = 0.05,
                 post: Callable[..., requests.Response] | None = None) -> None:
        self._key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        if not self._key:
            raise RuntimeError("TYPESAFE_API_KEY is not set")
        self.endpoint, self.model, self.timeout_s = endpoint, model, timeout_s
        self.noul_uncertainty = noul_uncertainty
        self._post = post or requests.post
        self.last_elapsed_ms = float("nan")

    def ask(self, state: dict, questions: Sequence[PhysicalQuestion]) -> list[Answer]:
        """One request, no retries: a reflex never waits for a retry."""
        start = time.perf_counter()
        try:
            r = self._post(self.endpoint, json=build_request(state, questions, self.model),
                           headers={"authorization": f"Bearer {self._key}"}, timeout=self.timeout_s)
            r.raise_for_status()
            return parse_answers(r.json(), questions, self.noul_uncertainty)
        finally:
            self.last_elapsed_ms = (time.perf_counter() - start) * 1000
