import json
from dataclasses import asdict

import pytest

pytest.importorskip("gymnasium")

from toddler.learn import generations as G  # noqa: E402
from toddler.learn import ppo  # noqa: E402

SMALL = dict(total_steps=2048, rollout=1024, epochs=1, minibatch=256)


def _rec(gen, tid, parents=()):
    return G.ToddlerRecord(gen, tid, "cartpole", {"seed": 1}, 2048, list(parents), [0.1, 0.2], {"device": "cpu"})


def test_save_load_round_trip_with_hash_check(tmp_path):
    reg = G.Registry(tmp_path)
    net, _ = ppo.train("cartpole", ppo.PPOConfig(seed=1, **SMALL))
    reg.save(net, _rec("gen-0", "t1"))
    net2, rec = reg.load("gen-0", "t1")
    assert all((a == b).all() for a, b in zip(net.state_dict().values(), net2.state_dict().values()))
    assert rec.business["invoicing_entity"] == "VirtualV Holding B.V."
    (tmp_path / "gen-0" / "t1" / "weights.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="sha256"):
        reg.load("gen-0", "t1")


def test_lineage_and_generation_listing(tmp_path):
    reg = G.Registry(tmp_path)
    net, _ = ppo.train("cartpole", ppo.PPOConfig(seed=2, **SMALL))
    reg.save(net, _rec("gen-0", "a"))
    reg.save(net, _rec("gen-1", "b", parents=["gen-0/a"]))
    reg.save(net, _rec("gen-2", "c", parents=["gen-1/b"]))
    assert reg.lineage("gen-2", "c") == ["gen-1/b", "gen-0/a"]
    assert [r.toddler_id for r in reg.generation("gen-1")] == ["b"]


def test_checkpoints_are_written_during_training(tmp_path):
    reg = G.Registry(tmp_path)
    ppo.train("cartpole", ppo.PPOConfig(seed=3, total_steps=4096, rollout=1024, epochs=1, minibatch=256),
              checkpoint=reg.checkpoint_fn("gen-0", "x"), checkpoint_every=2)
    assert len(list((tmp_path / "gen-0" / "x" / "checkpoints").glob("step_*.pt"))) == 2


def test_promotion_rule():
    better = [0.6, 0.7, 0.8, 0.9, 1.0]
    worse = [0.1, 0.2, 0.2, 0.3, 0.1]
    assert G.decide_promotion(better, worse).promote
    assert not G.decide_promotion(worse, better).promote
    with pytest.raises(ValueError):
        G.decide_promotion([0.9], worse)


def test_frozen_reference_refuses_changed_fingerprint(tmp_path):
    from toddler import quotients

    reg = G.Registry(tmp_path)
    for tid in ("a", "b"):
        (tmp_path / "gen-0" / tid).mkdir(parents=True)
        (tmp_path / "gen-0" / tid / "meta.json").write_text(json.dumps(asdict(_rec("gen-0", tid))))
    fp = quotients.fingerprint(["cartpole"], (10000, 10001), [22.0], [475.0])
    reg.freeze_reference("gen-0", fp)
    assert reg.reference("gen-0", fp) == pytest.approx([0.15, 0.15])
    with pytest.raises(ValueError):
        reg.reference("gen-0", quotients.fingerprint(["cartpole"], (10000, 10001), [23.0], [475.0]))


def test_frozen_reference_is_never_overwritten_silently(tmp_path):
    from toddler import quotients

    reg = G.Registry(tmp_path)
    for tid in ("a", "b"):
        (tmp_path / "gen-0" / tid).mkdir(parents=True)
        (tmp_path / "gen-0" / tid / "meta.json").write_text(json.dumps(asdict(_rec("gen-0", tid))))
    fp = quotients.fingerprint(["cartpole"], (10000,), [22.0], [475.0])
    other = quotients.fingerprint(["cartpole"], (10000,), [22.5], [475.0])
    path = reg.freeze_reference("gen-0", fp)
    before = path.read_text()
    assert reg.freeze_reference("gen-0", fp).read_text() == before      # same print: kept
    with pytest.raises(ValueError, match="frozen under another fingerprint"):
        reg.freeze_reference("gen-0", other)
    reg.freeze_reference("gen-0", other, refreeze=True)
    assert reg.reference("gen-0", other)


def test_records_from_a_newer_schema_load_with_extras_kept():
    d = asdict(_rec("gen-0", "t1"))
    d["future_field"] = 42
    rec = G.ToddlerRecord.from_dict(d)
    assert rec.extra == {"future_field": 42} and rec.toddler_id == "t1"
