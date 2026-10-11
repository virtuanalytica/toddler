"""G5 must reject wrong ancestry before any candidate can train."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from scripts import g5_parent_contract as C
from toddler.learn.multitask import MultiTaskNet
from toddler.learn.routing import TaskExpertRouter


def _fixture(monkeypatch):
    dims = {task: (4, 2) for task in C.NAV_TASKS}
    router = TaskExpertRouter({"a": MultiTaskNet(dims, 4),
                               "b": MultiTaskNet(dims, 4)},
                              {task: "a" for task in dims})
    births = {}
    for child in C.CHILD_IDS:
        ref = f"{C.GENERATION}/{child}"
        births[ref] = {"ref": ref, "weights_sha256": "a" * 64,
                       "inherits": [{"parent": f"G3-recombined/t500{child[-1]}",
                                     "parent_weights_sha256": "b" * 64}]}
    verdict = {"verdict": "survived", "_seq": 42,
               "evidence": {"promote": True, "protocol_sha256": "c" * 64,
                            "trial_sha256": ["d" * 64, "e" * 64]}}

    class FakeLedger:
        def __init__(self, root):
            pass

        def verify(self):
            return True, None

        def verdict(self, generation):
            return verdict

        def members(self, generation):
            return list(births.values())

        def birth(self, ref):
            return births[ref]

        def weights_of(self, ref):
            return "b" * 64

    class FakeRegistry:
        def __init__(self, root):
            pass

        def load(self, generation, child):
            return router, SimpleNamespace(weights_sha256="a" * 64,
                                           parents=[f"G3-recombined/t500{child[-1]}"],
                                           config={"route": router.route})

    monkeypatch.setattr(C, "Ledger", FakeLedger)
    monkeypatch.setattr(C, "Registry", FakeRegistry)
    return verdict, births, router


def test_verified_five_parent_contract(monkeypatch, tmp_path):
    _fixture(monkeypatch)
    contract = C.current_contract(tmp_path)
    assert [p["ref"] for p in contract["parents"]] == [f"G4-search/{c}" for c in C.CHILD_IDS]
    C.verify_contract(tmp_path, contract)
    contract["parents"][0]["weights_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="differs"):
        C.verify_contract(tmp_path, contract)


def test_rejects_unpromoted_or_mismatched_parent(monkeypatch, tmp_path):
    verdict, births, _ = _fixture(monkeypatch)
    verdict["evidence"]["promote"] = False
    with pytest.raises(ValueError, match="verdict"):
        C.current_contract(tmp_path)
    verdict["evidence"]["promote"] = True
    births["G4-search/t6003"]["weights_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="registry weights"):
        C.current_contract(tmp_path)


def test_expert_digest_changes_with_weights():
    layer = torch.nn.Linear(3, 2)
    before = C.module_digest(layer)
    with torch.no_grad():
        layer.weight[0, 0] += 1
    assert C.module_digest(layer) != before
