"""Jev client tests. Responses follow the documented TypeSafe System One schema (the same
schema virtualpc-jev-finance validates with zod); the transport is injected, no network."""

import pytest
import requests

from toddler import fastpath as fp
from toddler import jev

QS = fp.DEFAULT_QUESTIONS
CALM = fp.Sensors(10, 0.1, 2.0)


class _Resp:
    def __init__(self, body, status=200):
        self._body, self.status_code = body, status

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


def _client(body, status=200, record=None):
    def post(url, json, headers, timeout):
        if record is not None:
            record.update(url=url, json=json, timeout=timeout)
        return _Resp(body, status)
    return jev.HttpJevClient(api_key="test", post=post)


def _noul(**p):
    return {"model": "jev-latest", "usage": {"input_tokens": 1, "output_tokens": 1},
            "answers": {q.qid: {"type": "noul", "noul": p.get(q.qid, 0.02)} for q in QS}}


def test_request_uses_noul_questions_and_tight_timeout():
    rec = {}
    _client(_noul(), record=rec).ask({"t": 0}, QS)
    assert rec["json"]["questions"]["person_in_zone"]["type"] == "noul"
    assert rec["timeout"] == pytest.approx(0.02)


def test_calm_answers_continue():
    answers = _client(_noul()).ask({}, QS)
    assert fp.reflex(CALM, answers, elapsed_ms=5).command == "continue"


def test_choice_answer_uses_confidence_as_uncertainty():
    body = _noul()
    body["answers"]["collision_soon"] = {"type": "choice", "choice": "yes", "probabilities": {"yes": 0.2, "no": 0.8},
                                         "confidence": 0.5}
    a = {x.qid: x for x in _client(body).ask({}, QS)}["collision_soon"]
    assert a.probability == 0.2 and a.uncertainty == 0.5


def test_malformed_answer_raises_so_reflex_stops():
    body = _noul()
    body["answers"]["grasp_slip"] = {"type": "noul", "noul": 1.7}
    with pytest.raises(jev.JevResponseError):
        _client(body).ask({}, QS)
    assert fp.reflex(CALM, None, elapsed_ms=5).command == "stop"


def test_http_error_propagates():
    with pytest.raises(requests.HTTPError):
        _client(_noul(), status=529).ask({}, QS)


def test_missing_key_refused(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        jev.HttpJevClient()
