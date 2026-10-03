"""Own TypeSafe System One-compatible Jev server.

POST /v1/systemone with {"state", "questions", "model"} returns {"model", "answers", "usage"},
the exact schema validated by virtualpc-jev-finance (zod) and toddler/jev.py.

  noul   -> {"type": "noul", "noul": P(true)}
  choice -> {"type": "choice", "choice", "probabilities", "confidence"}
  score  -> {"type": "score", "score", "legend", "probabilities", "confidence"}

Confidence is the probability of the chosen option. An optional calibrator (calibrate.py)
maps raw noul probabilities to calibrated ones. A TTL cache keyed on (quantised state, question)
can let the reflex path answer within its budget, but only when the scene repeats: continuous
sensor states rarely repeat exactly, so floats are rounded to DEFAULT_CACHE_DECIMALS for the key
(None = exact keys). A live loop that needs every tick in budget also needs a local Jev.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Annotated, Literal, Union

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from jevserver.backend import Backend, LlamaCppBackend
from jevserver.keys import KeyStore

MODEL_NAME = os.environ.get("TODDLER_JEV_MODEL_NAME", "toddler-jev-qwen3.8-27b")


class Noul(BaseModel):
    type: Literal["noul"]
    instructions: str
    criteria: dict[str, str] | None = None


class Choice(BaseModel):
    type: Literal["choice"]
    instructions: str
    criteria: dict[str, str | None]


class Score(BaseModel):
    type: Literal["score"]
    instructions: str
    criteria: list[str]


Question = Annotated[Union[Noul, Choice, Score], Field(discriminator="type")]


class Request(BaseModel):
    state: object = None
    questions: dict[str, Question]
    model: str | None = None


class TTLCache:
    def __init__(self, ttl_s: float = 300.0, max_items: int = 50_000) -> None:
        self.ttl_s, self.max_items, self._d = ttl_s, max_items, {}

    def get(self, key: str):
        hit = self._d.get(key)
        if hit is None:
            return None
        if time.monotonic() - hit[0] < self.ttl_s:
            return hit[1]
        del self._d[key]               # expired: purge on read
        return None

    def put(self, key: str, value) -> None:
        if len(self._d) >= self.max_items:
            self._d.pop(next(iter(self._d)))
        self._d[key] = (time.monotonic(), value)


DEFAULT_CACHE_DECIMALS = 2
MAX_TEXT = 4000   # characters of state + instructions per question; longer input is refused


def quantise(state, decimals: int | None):
    """Round floats in the state so nearby sensor readings share a cache entry. Exact keys
    (decimals=None) almost never repeat for continuous sensors, so a live loop needs a
    quantised key, a local Jev, or both."""
    if decimals is None:
        return state
    if isinstance(state, float):
        return round(state, decimals) + 0.0      # + 0.0 turns -0.0 into 0.0, one key for both
    if isinstance(state, dict):
        return {k: quantise(v, decimals) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        return [quantise(v, decimals) for v in state]
    return state


def _key(state, qid: str, q: BaseModel) -> str:
    raw = json.dumps([state, qid, q.model_dump()], sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def create_app(backend: Backend | None = None, keys: KeyStore | None = None, calibrator=None,
               cache: TTLCache | None = None, cache_decimals: int | None = DEFAULT_CACHE_DECIMALS) -> FastAPI:
    backend = backend or LlamaCppBackend()
    keys = keys or KeyStore()
    cache = cache or TTLCache()
    app = FastAPI(title="Toddler Jev server", version="0.1")

    def auth(authorization: str = Header(default="")) -> str:
        token = authorization.removeprefix("Bearer ").strip()
        key_id = keys.verify(token) if token else None
        if not key_id:
            raise HTTPException(status_code=401, detail="invalid or revoked API key")
        return key_id

    @app.get("/health")
    def health():
        return {"ok": True, "model": MODEL_NAME}

    @app.post("/v1/systemone")
    def systemone(req: Request, _key_id: str = Depends(auth)):
        answers, tin, tout = {}, 0, 0
        state = quantise(req.state, cache_decimals)
        for qid, q in req.questions.items():
            if len(json.dumps(state, default=str)) + len(q.instructions) > MAX_TEXT:
                raise HTTPException(status_code=413, detail=f"{qid}: state + instructions too long")
            ck = _key(state, qid, q)
            cached = cache.get(ck)
            if cached is not None:
                answers[qid] = cached
                continue
            if isinstance(q, Noul):
                labels = ["yes", "no"]
            elif isinstance(q, Choice):
                labels = list(q.criteria.keys())
            else:
                labels = list(q.criteria)
            if len(labels) < 2:
                raise HTTPException(status_code=422, detail=f"{qid}: need at least two options")
            probs, a, b = backend.option_probs(state, q.instructions, labels)
            tin, tout = tin + a, tout + b
            if isinstance(q, Noul):
                p = probs["yes"]
                if calibrator is not None:
                    p = float(calibrator(p))
                ans = {"type": "noul", "noul": p}
            else:
                best = max(probs, key=probs.get)
                ans = {"type": "choice", "choice": best, "probabilities": probs, "confidence": probs[best]}
                if isinstance(q, Score):
                    ans = {"type": "score", "score": float(q.criteria.index(best)),
                           "legend": {str(i): c for i, c in enumerate(q.criteria)},
                           "probabilities": probs, "confidence": probs[best]}
            cache.put(ck, ans)
            answers[qid] = ans
        # Always report the model that actually answered, whatever the client asked for.
        return {"model": MODEL_NAME, "answers": answers,
                "usage": {"input_tokens": tin, "output_tokens": tout}}

    return app
