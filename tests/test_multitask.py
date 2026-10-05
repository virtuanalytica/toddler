import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("torch")

import torch  # noqa: E402

from toddler.learn import generations as G  # noqa: E402
from toddler.learn import multitask as M  # noqa: E402
from toddler.learn import ppo  # noqa: E402
from toddler.learn import tasks as T  # noqa: E402

SMALL = ppo.PPOConfig(rollout=512, epochs=1, minibatch=256)


def test_views_share_the_trunk_and_keep_their_own_heads():
    net = M.MultiTaskNet({"cartpole": (4, 2), "acrobot": (6, 3)})
    a, b = M.TaskView(net, "cartpole"), M.TaskView(net, "acrobot")
    shared = [p for p in a.parameters() if any(p is q for q in b.parameters())]
    assert shared == list(net.trunk.parameters())
    logits, value = a(torch.zeros(4))
    assert logits.shape == (2,) and value.shape == ()
    assert b(torch.zeros(6))[0].shape == (3,)


def test_training_updates_the_shared_trunk_from_every_task():
    torch.manual_seed(0)
    net, log = M.train_multitask(["cartpole", "acrobot"], steps_per_task=1024, block_steps=512, seed=3, base=SMALL)
    assert log.steps_per_task == {"cartpole": 1024, "acrobot": 1024} and log.blocks == 2
    with pytest.raises(ValueError):
        M.train_multitask(["cartpole"], steps_per_task=1000, block_steps=512, seed=3, base=SMALL)


def test_registry_round_trips_a_multitask_toddler(tmp_path):
    net = M.MultiTaskNet({"cartpole": (4, 2), "acrobot": (6, 3)})
    reg = G.Registry(tmp_path)
    rec = G.ToddlerRecord("G1", "t1", "multitask:cartpole+acrobot", {"seed": 1}, 2048, [], [0.1, 0.2], {"device": "cpu"})
    reg.save(net, rec)
    back, rec2 = reg.load("G1", "t1")
    assert isinstance(back, M.MultiTaskNet) and rec2.task.startswith("multitask:")
    assert all(torch.equal(x, y) for x, y in zip(net.state_dict().values(), back.state_dict().values()))


def test_sampled_evaluation_per_task():
    net = M.MultiTaskNet.for_tasks(["cartpole", "acrobot"])
    anchors = {t: T.random_anchor(t, T.EVAL_SEEDS[:2]) for t in ["cartpole", "acrobot"]}
    out = M.evaluate_multitask(net, ["cartpole", "acrobot"], anchors, seeds=T.EVAL_SEEDS[:2])
    assert set(out) == {"cartpole", "acrobot"} and all(len(v) == 2 for v in out.values())


def test_inherit_copies_trunk_and_shared_tasks_and_keeps_new_tasks_fresh():
    import torch

    from toddler.learn.multitask import MultiTaskNet

    torch.manual_seed(0)
    parent = MultiTaskNet({"a": (4, 2)}, hidden=8)
    child = MultiTaskNet({"a": (4, 2), "b": (6, 3)}, hidden=8)
    fresh_b = child.pi["b"].weight.clone()
    assert child.inherit(parent) == ["a"]
    for k, v in parent.state_dict().items():
        assert torch.equal(child.state_dict()[k], v)
    assert torch.equal(child.pi["b"].weight, fresh_b)


def test_inherit_refuses_mismatched_parents():
    import pytest

    from toddler.learn.multitask import MultiTaskNet

    with pytest.raises(ValueError):
        MultiTaskNet({"a": (4, 2)}, hidden=8).inherit(MultiTaskNet({"a": (4, 2)}, hidden=16))
    with pytest.raises(ValueError):
        MultiTaskNet({"b": (4, 2)}, hidden=8).inherit(MultiTaskNet({"a": (4, 2)}, hidden=8))
    with pytest.raises(ValueError):
        MultiTaskNet({"a": (5, 2)}, hidden=8).inherit(MultiTaskNet({"a": (4, 2)}, hidden=8))


def test_inherit_modes_shrink_perturb_and_trunk_only():
    import pytest
    import torch

    from toddler.learn.multitask import MultiTaskNet

    torch.manual_seed(1)
    parent = MultiTaskNet({"a": (4, 2)}, hidden=8)
    child = MultiTaskNet({"a": (4, 2), "b": (6, 3)}, hidden=8)
    fresh = {k: v.clone() for k, v in child.state_dict().items()}
    child.inherit(parent, mode="shrink_perturb", shrink=0.4, perturb=0.1)
    for k, v in parent.state_dict().items():
        assert torch.allclose(child.state_dict()[k], 0.4 * v + 0.1 * fresh[k])
    assert torch.equal(child.pi["b"].weight, fresh["pi.b.weight"])

    child2 = MultiTaskNet({"a": (4, 2)}, hidden=8)
    fresh2 = {k: v.clone() for k, v in child2.state_dict().items()}
    assert child2.inherit(parent, mode="trunk_only") == []
    assert torch.equal(child2.trunk[0].weight, parent.trunk[0].weight)
    assert torch.equal(child2.adapters["a"][0].weight, fresh2["adapters.a.0.weight"])
    with pytest.raises(ValueError):
        child2.inherit(parent, mode="mutate")
