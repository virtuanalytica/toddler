"""The successor stays pending until evidence and a trusted human signature agree."""

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch

from scripts.build_dashboard_data import collect
from scripts import import_g3_successor as importer
from toddler.learn.generations import Registry, ToddlerRecord
from toddler.learn.lineage import Ledger
from toddler.learn.multitask import MultiTaskNet
from toddler.learn.routing import TaskExpertRouter


def test_dashboard_keeps_g3_pending_and_does_not_invent_child_scores(tmp_path):
    review = Path(__file__).resolve().parents[1] / "docs/learn/G3_RECOMBINED_REVIEW.json"
    points = collect(tmp_path, review)
    assert len(points) == 1
    g3 = points[0]
    assert g3["verdict"] == "pending_review"
    assert g3["score_basis"].startswith("preregistered private")
    assert g3["tasks"]["unlockpickup"]["iqm"] == 0
    assert "values" not in g3["tasks"]["doorkey8"]


@pytest.mark.skipif(shutil.which("ssh-keygen") is None, reason="OpenSSH signing is unavailable")
def test_approval_requires_a_trusted_signature_for_the_exact_review(tmp_path):
    key = tmp_path / "reviewer"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True)
    signers = tmp_path / "allowed_signers"
    signers.write_text("operator " + key.with_suffix(".pub").read_text())
    payload = tmp_path / "approval.json"
    approval = {"candidate_generation": importer.GENERATION, "decision": "approve",
                "review_sha256": "a" * 64, "reviewed_by": "operator",
                "approved_at": datetime.now(timezone.utc).isoformat()}
    payload.write_text(json.dumps(approval, sort_keys=True) + "\n")
    subprocess.run(["ssh-keygen", "-Y", "sign", "-f", str(key), "-n", "toddler-lineage", str(payload)],
                   check=True, capture_output=True)
    signature = Path(str(payload) + ".sig")
    good = importer.verify_approval({"review_sha256": "a" * 64}, payload, signature, signers)
    assert good["reviewed_by"] == "operator"
    approval["review_sha256"] = "b" * 64
    payload.write_text(json.dumps(approval, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="signature"):
        importer.verify_approval({"review_sha256": "b" * 64}, payload, signature, signers)


def test_import_is_resumable_and_records_a_single_survivor(tmp_path, monkeypatch):
    monkeypatch.setattr(importer, "CHILDREN", ("t5001",))
    root, trial_root = tmp_path / "official", tmp_path / "trial"
    source, trial = Registry(root), Registry(trial_root)
    dims = {"cartpole": (4, 2)}
    torch.manual_seed(1)
    g2 = MultiTaskNet(dims)
    torch.manual_seed(2)
    other = MultiTaskNet(dims)
    parent_specs = (("G2", "t2001", g2), ("G3-sp", "t3001", other))
    ledger = Ledger(root)
    for generation, toddler_id, net in parent_specs:
        rec = ToddlerRecord(generation, toddler_id, "multitask:cartpole", {}, 1, [], [1.0], {})
        source.save(net, rec)
        ledger.born(f"{generation}/{toddler_id}", rec.weights_sha256, role="population",
                    code={}, data={}, budget={}, hardware={}, software={})
    ledger.selected("G2", "survived", {"promotion": True})
    route = {"cartpole": "G2"}
    candidate = TaskExpertRouter({"G2": g2, "G3-sp": other}, route)
    rec = ToddlerRecord(importer.GENERATION, "t5001", "task-router:cartpole", {"route": route}, 0,
                        ["G2/t2001", "G3-sp/t3001"], [1.0], {})
    trial.save(candidate, rec)
    evidence = {"official_root": root, "trial_root": trial_root,
                "review_sha256": "a" * 64,
                "review": {"candidate_weight_sha256": {"t5001": rec.weights_sha256},
                           "confirmations": [{"protocol_sha256": "b" * 64, "report_sha256": "c" * 64},
                                             {"protocol_sha256": "d" * 64, "report_sha256": "e" * 64}],
                           "known_limitations": ["test limitation"]},
                "report": {"promotion_vs_G2": {"promote": True},
                           "promotion_vs_random_control": {"promote": True},
                           "secret": {"candidate": {"aggregate_iqm": 0.95}}}}
    approval = {"reviewed_by": "operator", "approved_at": "2026-10-09T10:00:00Z",
                "approval_sha256": hashlib.sha256(b"approval").hexdigest(),
                "signature_sha256": hashlib.sha256(b"signature").hexdigest()}
    first = importer.apply(evidence, approval)
    second = importer.apply(evidence, approval)
    assert first == second == {"generation": importer.GENERATION, "verdict": "survived",
                               "members": 1, "ledger_verified": True}
    assert len([e for e in Ledger(root).log.entries if e.event_type == "selected"
                and e.details["generation"] == importer.GENERATION]) == 1
