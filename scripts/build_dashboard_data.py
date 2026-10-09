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


def collect(root: Path, review_path: Path | None = None) -> list[dict]:
    points, led = [], _ledger(root)
    for gdir in sorted(p for p in root.iterdir() if p.is_dir()):
        if gdir.name == "G3-recombined" and review_path is None:
            raise ValueError("G3-recombined needs its aggregate review reports for dashboard scores")
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
    return sorted(points, key=lambda p: p["trained_at"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=default_root())
    ap.add_argument("--out", default=str(REPO / "site" / "toddler" / "learning.json"))
    ap.add_argument("--review", type=Path, default=REPO / "docs/learn/G3_RECOMBINED_REVIEW.json")
    a = ap.parse_args()
    data = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "score": "normalised held-out score: 0 = random policy, 1 = task solved; interquartile mean over toddlers",
        "rule": "scores with different eval_mode or network are separate series and are never joined",
        "generations": collect(Path(a.root), a.review if a.review.exists() else None),
    }
    if a.review.exists():
        _, data["g3_review"] = _review_point(
            a.review, next((g["verdict"] for g in data["generations"]
                            if g["generation"] == "G3-recombined"), None))
    Path(a.out).write_text(json.dumps(data, indent=1) + "\n")
    print(f"{len(data['generations'])} generations -> {a.out}")


if __name__ == "__main__":
    main()
