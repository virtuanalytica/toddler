"""Build the data file behind the Toddler learning dashboard (site/toddler/learning.json).

Reads every generation in the registry (meta.json per toddler, reference.json where frozen) and
writes one point per generation: when it was trained, how (method, network, evaluation mode) and
the interquartile mean of the normalised held-out score per task over its toddlers. Scores from
different evaluation modes are never merged; the dashboard draws them as separate series.

Run: PYTHONPATH=. python3 scripts/build_dashboard_data.py [--root DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from toddler.learn.scoring import iqm

REPO = Path(__file__).resolve().parents[1]


def default_root() -> str:
    return os.environ.get("TODDLER_GENERATIONS_ROOT") or str(Path.home() / ".local/share/toddler/generations")


def _per_task(meta: dict) -> dict[str, float]:
    """Normalised held-out score per task for one toddler record."""
    cfg = meta.get("config", {})
    if cfg.get("per_task"):
        return {k: float(v) for k, v in cfg["per_task"].items()}
    return {meta["task"]: float(np.mean(meta["eval_scores"]))}


def _ledger(root: Path):
    """Selection verdicts and roles from the evolution ledger (toddler.learn.lineage), if present."""
    if not (root / "lineage.jsonl").exists():
        return None
    from toddler.learn import lineage

    return lineage.Ledger(root)


def _review_point(review_path: Path, official_verdict: str | None) -> tuple[dict, dict]:
    """Show only verified, aggregate review results; no private seeds or invented child scores."""
    packet = json.loads(review_path.read_text())
    if packet.get("candidate_generation") != "G3-recombined" or len(packet.get("confirmations", [])) != 2:
        raise ValueError("unexpected G3 review packet")
    reports = []
    protocols = []
    for item in packet["confirmations"]:
        protocol_path = REPO / item["protocol"]
        if hashlib.sha256(protocol_path.read_bytes()).hexdigest() != item["protocol_sha256"]:
            raise ValueError("G3 dashboard protocol hash mismatch")
        protocols.append(json.loads(protocol_path.read_text()))
        path = REPO / item["report"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["report_sha256"]:
            raise ValueError("G3 dashboard report hash mismatch")
        report = json.loads(path.read_text())
        if (report["protocol_sha256"] != item["protocol_sha256"]
                or report["routes"] != protocols[-1]["candidate_routes"]
                or not report["eligible_for_lineage_review"] or not item["eligible_for_lineage_review"]):
            raise ValueError("G3 dashboard cannot show a failed review as a successor")
        reports.append(report)
    if reports[0]["secret_seed_commitment"]["set_id"] == reports[1]["secret_seed_commitment"]["set_id"]:
        raise ValueError("G3 review reused a seed set")
    if protocols[0]["candidate_routes"] != protocols[1]["candidate_routes"]:
        raise ValueError("G3 review replication changed the route")
    report = reports[0]
    n = len(report["secret"]["candidate"]["per_child_mean"])
    tasks = {task: {"iqm": round(score, 4), "n": n}
             for task, score in report["secret"]["candidate"]["per_task_iqm"].items()}
    point = {"generation": "G3-recombined", "trained_at": report["secret_seed_commitment"]["created_at"],
             "toddlers": n, "network": "task-router", "method": "frozen-task-expert-route",
             "scale_rewards": True, "eval_mode": "sample", "parents": sorted(report["routes"]),
             "frozen_reference": False, "tasks": tasks,
             "iq_raw": round(float(np.mean([t["iqm"] for t in tasks.values()])), 4),
             "aggregate_iqm": round(report["secret"]["candidate"]["aggregate_iqm"], 4),
             "verdict": official_verdict or "pending_review", "control": False,
             "score_basis": "preregistered private set; aggregate only"}
    comparisons = [{"label": item["label"], "candidate": round(r["promotion_vs_G2"]["candidate_iqm"], 4),
                    "g2": round(r["promotion_vs_G2"]["reference_iqm"], 4),
                    "p_vs_g2": round(r["promotion_vs_G2"]["p_value"], 5),
                    "p_vs_random": round(r["promotion_vs_random_control"]["p_value"], 5)}
                   for item, r in zip(packet["confirmations"], reports)]
    return point, {"status": point["verdict"], "children": n, "comparisons": comparisons,
                   "limitation": "UnlockPickup unsolved on both private sets"}


def _g4_point(status_path: Path, audit_dir: Path, verdict: dict | None) -> tuple[dict, dict]:
    """Export only verified G4 cohort aggregates, never private seed rows."""
    status = json.loads(status_path.read_text())
    if (status.get("status") != "promoted" or status.get("generation") != "G4-search"
            or status.get("children") != 5 or status.get("ledger_verified") is not True
            or not verdict or verdict.get("verdict") != "survived"):
        raise ValueError("G4 dashboard requires a promoted status and survived official lineage")
    evidence = verdict.get("evidence") or {}
    if evidence.get("promote") is not True or len(evidence.get("trial_sha256", [])) != 2:
        raise ValueError("G4 lineage lacks two passing trial commitments")
    protocol_path = audit_dir / "protocol.json"
    protocol_bytes = protocol_path.read_bytes()
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    if protocol_hash != status.get("protocol_sha256") or protocol_hash != evidence.get("protocol_sha256"):
        raise ValueError("G4 frozen protocol hash mismatch")
    protocol = json.loads(protocol_bytes)
    manifest_path = Path(protocol["manifest"])
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != protocol["manifest_sha256"]:
        raise ValueError("G4 candidate manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if (status.get("cohort") != manifest.get("created_utc")
            or protocol.get("generation") != "G4-search"
            or len(protocol.get("children", [])) != 5
            or len(protocol.get("tasks", [])) != 9):
        raise ValueError("G4 cohort and frozen protocol disagree")
    trials = []
    for index, expected_hash in enumerate(evidence["trial_sha256"], start=1):
        path = audit_dir / f"trial-{index}.json"
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise ValueError("G4 private trial hash mismatch")
        trial = json.loads(data)
        if (trial.get("frozen_plan_sha256") != protocol_hash
                or trial.get("child_ids") != protocol["children"]
                or trial.get("candidate_weights") != protocol["candidate_weights"]
                or set(trial.get("candidate", {})) != set(protocol["tasks"])):
            raise ValueError("G4 trial does not match the frozen cohort")
        for task in protocol["tasks"]:
            values = trial["candidate"][task]
            if len(values) != 5 or not all(np.isfinite(v) for v in values):
                raise ValueError("G4 private aggregate has invalid child values")
        trials.append(trial)
    if trials[0]["seed_set_id"] == trials[1]["seed_set_id"]:
        raise ValueError("G4 private trials reused a seed set")
    if not all(row.get("passed") is True for row in evidence.get("trials", [])):
        raise ValueError("G4 lineage lacks two passing comparisons")
    # The first confirmation supplies the plotted per-task aggregates; the
    # separate second confirmation is reported without exposing seed IDs.
    tasks = {task: {"iqm": round(iqm(np.asarray(trials[0]["candidate"][task])), 4), "n": 5}
             for task in protocol["tasks"]}
    created = datetime.strptime(manifest["created_utc"].split("-", 1)[0], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    point = {"generation": "G4-search", "trained_at": created.isoformat().replace("+00:00", "Z"),
             "toddlers": 5, "network": "task-router", "method": "selected-inherited-expert",
             "scale_rewards": True, "eval_mode": protocol["eval_mode"],
             "parents": protocol["parents"], "frozen_reference": False,
             "tasks": tasks, "iq_raw": round(float(np.mean([v["iqm"] for v in tasks.values()])), 4),
             "verdict": "survived", "control": False,
             "score_basis": "private matched trial 1 aggregate; separate trial 2 confirmation"}
    comparisons = [{"label": f"Private toets {i}",
                    "candidate": round(row["vs_ancestor"]["candidate_mean"], 4),
                    "g3": round(row["vs_ancestor"]["reference_mean"], 4),
                    "p_vs_g3": row["vs_ancestor"]["p_value"],
                    "p_vs_control": row["vs_control"]["p_value"]}
                   for i, row in enumerate(evidence["trials"], start=1)]
    return point, {"status": "survived", "children": 5, "comparisons": comparisons,
                   "scope": "navigation only; cognition, software and physical tasks unpromoted"}


def collect(root: Path, review_path: Path | None = None,
            g4_status: Path | None = None, g4_audit: Path | None = None) -> list[dict]:
    if bool(g4_status) != bool(g4_audit):
        raise ValueError("G4 dashboard requires both status and audit directory")
    points, led = [], _ledger(root)
    if g4_status and (not led or led.verify()[0] is not True):
        raise ValueError("G4 dashboard requires a verified official lineage ledger")
    for gdir in sorted(p for p in root.iterdir() if p.is_dir()):
        if gdir.name == "G3-recombined" and review_path is None:
            raise ValueError("G3-recombined needs its aggregate review reports for dashboard scores")
        if gdir.name == "G4-search" and not g4_status:
            raise ValueError("G4-search needs verified private aggregate trials for dashboard scores")
        metas = [json.loads(m.read_text()) for m in sorted(gdir.glob("*/meta.json"))]
        if not metas:
            continue
        cfg = metas[0].get("config", {})
        scores: dict[str, list[float]] = defaultdict(list)
        for m in metas:
            for task, s in _per_task(m).items():
                scores[task].append(s)
        tasks = {t: {"iqm": round(iqm(np.asarray(v)), 4), "n": len(v),
                     "values": [round(x, 4) for x in v]} for t, v in sorted(scores.items())}
        ref = gdir / "reference.json"
        points.append({
            "generation": gdir.name,
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(max(m["created_at"] for m in metas))),
            "toddlers": len(metas),
            "network": ("task-router" if str(metas[0].get("task", "")).startswith("task-router:")
                        else "multitask" if str(metas[0].get("task", "")).startswith("multitask:")
                        else "single-task"),
            "method": cfg.get("method", "ppo"),
            "scale_rewards": cfg.get("scale_rewards", False),
            "eval_mode": cfg.get("eval_mode", "greedy"),
            "parents": sorted({p for m in metas for p in m.get("parents", [])}),
            "frozen_reference": ref.exists(),
            "tasks": tasks,
            "iq_raw": round(float(np.mean([t["iqm"] for t in tasks.values()])), 4),
            "verdict": (led.verdict(gdir.name) or {}).get("verdict") if led else None,
            "control": bool(led) and bool(led.members(gdir.name)) and all(
                m["role"] == "control" for m in led.members(gdir.name)),
            "score_basis": "public evaluation seeds",
        })
    if review_path is not None:
        point, _ = _review_point(review_path, (led.verdict("G3-recombined") or {}).get("verdict") if led else None)
        points = [row for row in points if row["generation"] != point["generation"]]
        points.append(point)
    if g4_status and g4_audit:
        point, _ = _g4_point(g4_status, g4_audit, led.verdict("G4-search"))
        points = [row for row in points if row["generation"] != point["generation"]]
        points.append(point)
    return sorted(points, key=lambda p: p["trained_at"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=default_root())
    ap.add_argument("--out", default=str(REPO / "site" / "toddler" / "learning.json"))
    ap.add_argument("--review", type=Path, default=REPO / "docs/learn/G3_RECOMBINED_REVIEW.json")
    ap.add_argument("--g4-status", type=Path, help="owner-only G4 promotion status")
    ap.add_argument("--g4-audit", type=Path, help="directory with frozen G4 protocol and two private trials")
    a = ap.parse_args()
    root = Path(a.root)
    data = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "score": "normalised held-out score: 0 = random policy, 1 = task solved; interquartile mean over toddlers",
        "rule": "scores with different eval_mode or network are separate series and are never joined",
        "generations": collect(root, a.review if a.review.exists() else None, a.g4_status, a.g4_audit),
    }
    if a.review.exists():
        _, data["g3_review"] = _review_point(
            a.review, next((g["verdict"] for g in data["generations"]
                            if g["generation"] == "G3-recombined"), None))
    if a.g4_status and a.g4_audit:
        data["g4_review"] = _g4_point(a.g4_status, a.g4_audit, _ledger(root).verdict("G4-search"))[1]
    Path(a.out).write_text(json.dumps(data, indent=1) + "\n")
    print(f"{len(data['generations'])} generations -> {a.out}")


if __name__ == "__main__":
    main()
