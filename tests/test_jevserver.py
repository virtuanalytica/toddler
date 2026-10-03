"""Jev server tests: contract, auth, cache, and round trip through toddler/jev.py.
The backend is a deterministic stand-in; the live model is exercised by the smoke run."""

import pytest
from fastapi.testclient import TestClient

from jevserver.app import TTLCache, create_app
from jevserver.keys import KeyStore
from toddler import fastpath as fp
from toddler import jev


class FixedBackend:
    def __init__(self):
        self.calls = 0

    def option_probs(self, state, instructions, options):
        self.calls += 1
        if options == ["yes", "no"]:
            return {"yes": 0.03, "no": 0.97}, 10, 1
        p = 1.0 / len(options)
        return {o: p for o in options}, 10, 1


@pytest.fixture()
def setup(tmp_path):
    store = KeyStore(tmp_path / "keys.json")
    key_id, key = store.create("test")
    backend = FixedBackend()
    client = TestClient(create_app(backend=backend, keys=store, cache=TTLCache()))
    return client, store, key_id, key, backend


def test_key_file_stores_only_hash(setup, tmp_path):
    _, _, _, key, _ = setup
    text = (tmp_path / "keys.json").read_text()
    assert key not in text and (tmp_path / "keys.json").stat().st_mode & 0o777 == 0o600


def test_auth_required_and_revocation(setup):
    client, store, key_id, key, _ = setup
    body = {"state": {}, "questions": {"q": {"type": "noul", "instructions": "x"}}}
    assert client.post("/v1/systemone", json=body).status_code == 401
    assert client.post("/v1/systemone", json=body, headers={"authorization": f"Bearer {key}"}).status_code == 200
    store.revoke(key_id)
    assert client.post("/v1/systemone", json=body, headers={"authorization": f"Bearer {key}"}).status_code == 401


def test_contract_shapes(setup):
    client, _, _, key, _ = setup
    body = {"state": {"a": 1}, "questions": {
        "n": {"type": "noul", "instructions": "x"},
        "c": {"type": "choice", "instructions": "x", "criteria": {"left": None, "right": None}},
        "s": {"type": "score", "instructions": "x", "criteria": ["low", "mid", "high"]}}}
    out = client.post("/v1/systemone", json=body, headers={"authorization": f"Bearer {key}"}).json()
    assert set(out) == {"model", "answers", "usage"}
    assert out["answers"]["n"] == {"type": "noul", "noul": 0.03}
    assert out["answers"]["c"]["type"] == "choice" and 0 <= out["answers"]["c"]["confidence"] <= 1
    assert out["answers"]["s"]["legend"] == {"0": "low", "1": "mid", "2": "high"}


def test_cache_avoids_second_model_call(setup):
    client, _, _, key, backend = setup
    body = {"state": {"x": 1}, "questions": {"q": {"type": "noul", "instructions": "x"}}}
    h = {"authorization": f"Bearer {key}"}
    client.post("/v1/systemone", json=body, headers=h)
    client.post("/v1/systemone", json=body, headers=h)
    assert backend.calls == 1


def test_round_trip_with_toddler_client(setup):
    client, _, _, key, _ = setup

    def post(url, json, headers, timeout):
        return client.post("/v1/systemone", json=json, headers=headers)

    c = jev.HttpJevClient(api_key=key, endpoint="http://testserver/v1/systemone", post=post)
    answers = c.ask({"t": 0}, fp.DEFAULT_QUESTIONS)
    assert fp.reflex(fp.Sensors(10, 0.1, 2.0), answers, elapsed_ms=5).command == "continue"
