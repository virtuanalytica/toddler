import hashlib
import json

import pytest

from scripts.build_dashboard_data import _g4_point


def _write(path, data):
    path.write_text(json.dumps(data))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_g4_dashboard_exports_only_verified_aggregates(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest_hash = _write(manifest, {"created_utc": "20261010T200500Z-test"})
    children = [f"t600{i}" for i in range(1, 6)]
    tasks = ["cartpole", "acrobot", "empty5", "doorkey5", "doorkey8",
             "unlock", "unlockpickup", "keycorridor3", "lavacross9"]
    protocol = {"generation": "G4-search", "manifest": str(manifest),
                "manifest_sha256": manifest_hash, "children": children,
                "tasks": tasks, "candidate_weights": [f"sha-{i}" for i in range(5)],
                "parents": [f"G3-recombined/t500{i}" for i in range(1, 6)],
                "eval_mode": "sample"}
    protocol_hash = _write(tmp_path / "protocol.json", protocol)
    hashes = []
    for index in (1, 2):
        trial = {"frozen_plan_sha256": protocol_hash, "child_ids": children,
                 "candidate_weights": protocol["candidate_weights"],
                 "seed_set_id": f"sealed-{index}",
                 "candidate": {task: [float(index)] * 5 for task in tasks},
                 "per_seed": "PRIVATE_SENTINEL"}
        hashes.append(_write(tmp_path / f"trial-{index}.json", trial))
    status = tmp_path / "status.json"
    _write(status, {"status": "promoted", "generation": "G4-search", "children": 5,
                    "ledger_verified": True, "cohort": "20261010T200500Z-test",
                    "protocol_sha256": protocol_hash})
    comparison = {"passed": True, "vs_ancestor": {"candidate_mean": 1.0,
                    "reference_mean": .9, "p_value": .001},
                  "vs_control": {"p_value": .001}}
    verdict = {"verdict": "survived", "evidence": {"promote": True,
               "protocol_sha256": protocol_hash, "trial_sha256": hashes,
               "trials": [comparison, comparison]}}
    point, review = _g4_point(status, tmp_path, verdict)
    assert point["generation"] == "G4-search" and point["verdict"] == "survived"
    assert point["tasks"]["unlockpickup"] == {"iqm": 1.0, "n": 5}
    assert review["children"] == 5 and len(review["comparisons"]) == 2
    assert "PRIVATE_SENTINEL" not in json.dumps((point, review))

    (tmp_path / "trial-2.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        _g4_point(status, tmp_path, verdict)
