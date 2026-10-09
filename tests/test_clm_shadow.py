import math
import json

import pytest
import requests

from toddler import clm, fastpath as fp, reflex_shadow


class Response:
    def __init__(self, body, status=200):
        self.body, self.status = body, status

    def json(self):
        return self.body

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(str(self.status))


def body(probability=0.01):
    return {"model": "clm-latest", "answers": {q.qid: {"type": "noul", "noul": probability}
                                                for q in fp.DEFAULT_QUESTIONS}}


def test_clm_uses_local_systemone_contract_without_key():
    seen = {}

    def post(url, json, headers, timeout):
        seen.update(url=url, json=json, headers=headers, timeout=timeout)
        return Response(body())

    client = clm.HttpClmClient(post=post, api_key="")
    answers = client.ask({"grasp": "stable"}, fp.DEFAULT_QUESTIONS)
    assert seen["url"] == clm.DEFAULT_ENDPOINT
    assert seen["json"]["questions"]["grasp_slip"]["type"] == "noul"
    assert seen["timeout"] == 0.02 and seen["headers"] == {}
    assert len(answers) == 4 and client.last_elapsed_ms >= 0


def test_clm_rejects_bad_payload_and_http_failure():
    invalid = body()
    del invalid["answers"]["collision_soon"]
    with pytest.raises(ValueError):
        clm.HttpClmClient(post=lambda *a, **kw: Response(invalid)).ask({}, fp.DEFAULT_QUESTIONS)
    with pytest.raises(requests.HTTPError):
        clm.HttpClmClient(post=lambda *a, **kw: Response(body(), 503)).ask({}, fp.DEFAULT_QUESTIONS)
    with pytest.raises(ValueError):
        clm.HttpClmClient(timeout_s=0.1)


class Client:
    def __init__(self, p=None, error=False):
        self.p, self.error, self.calls = p, error, 0

    def ask(self, state, questions):
        self.calls += 1
        if self.error:
            raise TimeoutError()
        return [fp.Answer(q.qid, self.p, 0.05) for q in questions]


def test_shadow_never_changes_jev_decision():
    jev, clm_client = Client(0.01), Client(0.95)
    result = reflex_shadow.compare({}, fp.Sensors(10, 0.1, 2), jev, clm_client)
    assert result.jev.command == "continue"
    assert result.clm.command == "stop"
    assert jev.calls == clm_client.calls == 1


def test_shadow_failures_and_hard_limits():
    jev, clm_client = Client(0.01), Client(error=True)
    result = reflex_shadow.compare({}, fp.Sensors(10, 0.1, 2), jev, clm_client)
    assert result.jev.command == "continue" and result.clm.command == "stop"
    assert result.clm_error == "TimeoutError"
    result = reflex_shadow.compare({}, fp.Sensors(200, 0.1, 2), jev, clm_client)
    assert result.jev.command == result.clm.command == "stop"
    assert jev.calls == clm_client.calls == 1


@pytest.mark.parametrize("sensors,elapsed", [
    (fp.Sensors(math.nan, 0.1, 2), 5),
    (fp.Sensors(10, math.inf, 2), 5),
    (fp.Sensors(10, 0.1, 2), math.nan),
    (fp.Sensors(10, 0.1, 2), -1),
])
def test_non_finite_sensor_or_bad_clock_fails_closed(sensors, elapsed):
    answers = [fp.Answer(q.qid, 0.01, 0.05) for q in fp.DEFAULT_QUESTIONS]
    assert fp.reflex(sensors, answers, elapsed).command == "stop"


@pytest.mark.parametrize("probability,uncertainty", [(math.nan, 0.0), (0.0, math.inf),
                                                     (0.0, math.nan)])
def test_non_finite_answer_fails_closed(probability, uncertainty):
    answers = [fp.Answer(q.qid, probability, uncertainty) for q in fp.DEFAULT_QUESTIONS]
    assert fp.reflex(fp.Sensors(10, 0.1, 2), answers, 5).command == "stop"


def test_private_shadow_cli_emits_no_raw_scene(tmp_path):
    from scripts.compare_clm_reflex import run

    source, output = tmp_path / "scenes.jsonl", tmp_path / "results.jsonl"
    source.write_text(json.dumps({"state": {"secret": "do-not-log"},
                                  "sensors": {"force_n": 10, "speed_m_s": 0.1,
                                              "person_distance_m": 2}}) + "\n")
    report = run(source, output, Client(0.01), Client(0.95))
    assert report["items"] == report["paired"] == report["disagreements"] == 1
    assert "do-not-log" not in output.read_text()
    assert output.stat().st_mode & 0o777 == 0o600


def test_shadow_cli_rejects_mock_clm(monkeypatch):
    from scripts.compare_clm_reflex import clm_ready

    monkeypatch.setattr("scripts.compare_clm_reflex.requests.get",
                        lambda *a, **kw: Response({"ok": True, "embedder": True, "mock": True}))
    with pytest.raises(RuntimeError, match="mock"):
        clm_ready(clm.DEFAULT_ENDPOINT)
