import pytest

from toddler.learn import lineage as L

CODE = {"commit": "abc", "dirty": False}
DATA = {"tasks": {"cartpole": "CartPole-v1"}}


def _born(led, ref, sha, parents=(), role="population"):
    inh = [L.Inheritance(p, led.weights_of(p), ("trunk",)) for p in parents]
    return led.born(ref, sha, role=role, inherits=inh, code=CODE, data=DATA, budget={"steps": 1},
                    hardware={"cpu": "x"}, software={"torch": "2"})


@pytest.fixture
def led(tmp_path):
    led = L.Ledger(tmp_path)
    _born(led, "G1/t1", "s1")
    _born(led, "G1/t2", "s2")
    _born(led, "G2/t3", "s3", parents=["G1/t1"])
    _born(led, "G3/t4", "s4", parents=["G2/t3", "G1/t2"])
    return led


def test_ancestry_and_descendants_walk_the_whole_tree(led):
    assert [a["parent"] for a in led.ancestry("G3/t4")] == ["G2/t3", "G1/t2", "G1/t1"]
    assert led.descendants("G1/t1") == ["G2/t3", "G3/t4"]


def test_ledger_persists_and_its_hash_chain_verifies(led, tmp_path):
    again = L.Ledger(tmp_path)
    assert again.verify() == (True, None)
    assert again.birth("G2/t3")["inherits"][0]["parent_weights_sha256"] == "s1"


def test_inheritance_from_a_different_parent_model_is_refused(led):
    with pytest.raises(ValueError):
        led.born("G2/t9", "s9", role="population", inherits=[L.Inheritance("G1/t1", "not-s1", ("trunk",))],
                 code=CODE, data=DATA, budget={}, hardware={}, software={})


def test_double_birth_and_unknown_role_are_refused(led):
    with pytest.raises(ValueError):
        _born(led, "G1/t1", "s1")
    with pytest.raises(ValueError):
        _born(led, "G1/t8", "s8", role="mutant")


def test_only_survivors_are_listed_and_a_later_verdict_supersedes(led):
    led.selected("G1", "survived", {"why": "reference"})
    led.selected("G2", "extinct", {"why": "no gain"})
    led.selected("G2", "survived", {"why": "re-evaluated"})
    led.selected("G3", "control", {})
    assert led.surviving_generations() == ["G1", "G2"]
    with pytest.raises(ValueError):
        led.selected("G9", "survived", {})


def test_transmorphed_agent_traces_back_to_every_toddler_foundation(led):
    led.selected("G1", "survived", {})
    led.derived_agent("planner", "G3/t4", "route planning", "operator", weights_sha256="p1")
    led.derived_agent("reader", "G1/t2", "reading", "operator", weights_sha256="r1")
    led.transmorphed("planner-reader", [L.Inheritance("agent:planner", "p1", ("trunk",)),
                                        L.Inheritance("agent:reader", "r1", ("adapter:text",))],
                     ["planning", "reading"], "pr1", "cross-skill", "operator", CODE, DATA)
    tr = led.trace_agent("planner-reader")
    assert [a["agent_id"] for a in tr["chain"]] == ["planner-reader", "planner", "reader"]
    assert set(tr["foundations"]) == {"G3/t4", "G1/t2"}
    assert tr["verdicts"]["G1"]["verdict"] == "survived" and tr["verdicts"]["G3"] is None


def test_transmorphosis_requires_known_sources_with_matching_weights(led):
    led.derived_agent("a", "G1/t1", "x", "operator", weights_sha256="a1")
    with pytest.raises(KeyError):
        led.transmorphed("b", [L.Inheritance("agent:ghost", "g", ())], [], "b1", "x", "operator", CODE, DATA)
    with pytest.raises(ValueError):
        led.transmorphed("b", [L.Inheritance("agent:a", "wrong", ())], [], "b1", "x", "operator", CODE, DATA)
    with pytest.raises(ValueError):
        led.transmorphed("b", [L.Inheritance("agent:a", "a1", ())], [], "b1", "x", " ", CODE, DATA)


def test_credentials_never_enter_the_ledger(led):
    with pytest.raises(ValueError):
        led.born("G1/t7", "s7", role="population", code=CODE, data={"api_key": "x"}, budget={},
                 hardware={}, software={})


def test_code_version_reports_the_repo_commit():
    v = L.code_version()
    assert len(v["commit"]) == 40 and isinstance(v["dirty"], bool)
