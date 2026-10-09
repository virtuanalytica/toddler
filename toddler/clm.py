"""Local CLM SystemOne client for reflex shadow trials.

This transport implements the same ``ask`` protocol as JEV. Its fixed noul margin
is a comparison convention, *not* a calibrated physical-risk interval. No caller
may use it to actuate a robot until an independent safety gate is passed.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Sequence

import requests

from toddler.fastpath import Answer, PhysicalQuestion
from toddler.jev import build_request, parse_answers

DEFAULT_ENDPOINT = "http://127.0.0.1:8700/v1/systemone"
DEFAULT_MODEL = "clm-latest"


class HttpClmClient:
    def __init__(self, endpoint: str = DEFAULT_ENDPOINT, model: str = DEFAULT_MODEL,
                 api_key: str | None = None, timeout_s: float = 0.02,
                 noul_uncertainty: float = 0.05,
                 post: Callable[..., requests.Response] | None = None) -> None:
        if not 0 < timeout_s <= 0.02:
            raise ValueError("CLM shadow timeout must fit the 20 ms reflex budget")
        if not 0 <= noul_uncertainty <= 1:
            raise ValueError("noul_uncertainty must be in [0, 1]")
        self.endpoint = endpoint
        self.model = model
        self.timeout_s = timeout_s
        self.noul_uncertainty = noul_uncertainty
        self._key = api_key if api_key is not None else os.environ.get("CLM_API_KEY", "")
        self._post = post or requests.post
        self.last_elapsed_ms = float("nan")

    def ask(self, state: dict, questions: Sequence[PhysicalQuestion]) -> list[Answer]:
        start = time.perf_counter()
        headers = {"authorization": f"Bearer {self._key}"} if self._key else {}
        try:
            response = self._post(self.endpoint, json=build_request(state, questions, self.model),
                                  headers=headers, timeout=self.timeout_s)
            response.raise_for_status()
            return parse_answers(response.json(), questions, self.noul_uncertainty)
        finally:
            self.last_elapsed_ms = (time.perf_counter() - start) * 1000
