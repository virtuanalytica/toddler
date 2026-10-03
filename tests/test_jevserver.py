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


def test_backend_reads_both_llama_formats_and_avoids_prefix_double_count():
    from jevserver.backend import _candidates

    new = {"top_logprobs": [{"token": "Yes", "logprob": -0.1}, {"token": "no", "logprob": -2.3}]}
    old = {"top_probs": [{"token": "yes", "prob": 0.9}]}
    assert _candidates(new)[0][0] == "Yes" and abs(_candidates(new)[0][1] - 0.905) < 1e-3
    assert _candidates(old) == [("yes", 0.9)]


class _FakeResp:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


def test_backend_label_matching_and_zero_mass(monkeypatch):
    from jevserver import backend as be

    body = {"completion_probabilities": [{"top_logprobs": [{"token": "low", "logprob": -0.2},
                                                           {"token": "lower", "logprob": -2.0}]}]}
    monkeypatch.setattr(be.requests, "post", lambda *a, **k: _FakeResp(body))
    probs, _, _ = be.LlamaCppBackend().option_probs({}, "x", ["low", "lower"])
    assert probs["low"] > probs["lower"]          # "lower" is not counted as "low"
    body["completion_probabilities"][0]["top_logprobs"] = [{"token": "banana", "logprob": -0.1}]
    with pytest.raises(RuntimeError):
        be.LlamaCppBackend().option_probs({}, "x", ["yes", "no"])


def test_server_reports_its_own_model_and_refuses_huge_input(setup):
    client, _, _, key, _ = setup
    h = {"authorization": f"Bearer {key}"}
    out = client.post("/v1/systemone", json={"model": "jev-latest", "state": {},
                                             "questions": {"q": {"type": "noul", "instructions": "x"}}}, headers=h).json()
    assert out["model"] != "jev-latest"
    big = {"state": {"t": "x" * 5000}, "questions": {"q": {"type": "noul", "instructions": "x"}}}
    assert client.post("/v1/systemone", json=big, headers=h).status_code == 413


def test_quantised_cache_key_shares_nearby_states(tmp_path):
    from jevserver.app import quantise

    assert quantise({"d": 0.30001, "v": [1.00004]}, 2) == quantise({"d": 0.29999, "v": [0.99996]}, 2)


def test_ttl_cache_expires(monkeypatch):
    import jevserver.app as app_mod

    now = [1000.0]
    monkeypatch.setattr(app_mod.time, "monotonic", lambda: now[0])
    c = TTLCache(ttl_s=300.0)
    c.put("k", 1)
    now[0] += 299.0
    assert c.get("k") == 1
    now[0] += 2.0
    assert c.get("k") is None


def test_calibration_metrics_on_hand_computed_case():
    import numpy as np

    from jevserver import calibrate as cal

    p = np.array([0.9, 0.1, 0.6, 0.4])
    y = np.array([1.0, 0.0, 0.0, 1.0])
    ix = np.array([0, 0, 1, 1])
    m = cal.metrics(p, y, ix)
    assert m["brier"] == pytest.approx((0.01 + 0.01 + 0.36 + 0.36) / 4, abs=1e-4)
    assert m["log_loss"] == pytest.approx(-(np.log(0.9) * 2 + np.log(0.4) * 2) / 4, abs=1e-4)
    assert m["accuracy_at_0_5"] == 0.5 and m["piqa_pair_accuracy"] == 0.5
    # bins [0.1,0.2): |0.1-0|, [0.4,0.5): |0.4-1|, [0.6,0.7): |0.6-0|, [0.9,1]: |0.9-1|, each weight 1/4
    assert m["ece"] == pytest.approx((0.1 + 0.6 + 0.6 + 0.1) / 4, abs=1e-4)
    assert cal.ece(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 0.0


def test_expired_entries_are_purged_and_negative_zero_shares_a_key(monkeypatch):
    import jevserver.app as app_mod
    from jevserver.app import quantise

    now = [0.0]
    monkeypatch.setattr(app_mod.time, "monotonic", lambda: now[0])
    c = TTLCache(ttl_s=10.0)
    c.put("k", 1)
    now[0] = 11.0
    assert c.get("k") is None and "k" not in c._d
    import json

    assert json.dumps(quantise({"x": -0.0001}, 2)) == json.dumps(quantise({"x": 0.0}, 2))   # "0.0", not "-0.0"
