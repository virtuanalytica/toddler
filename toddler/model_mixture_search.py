"""Search candidate LLM task routes using public development measurements only.

Inputs are newly measured, anti-contamination development summaries. Historical
public full-suite rows and private audit results are deliberately ineligible.
The output is a research candidate, never a promotion or a serving config.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

SCHEMA = "toddler-mom-public-dev/v1"
POLICY = "anti_contamination_public_development"
MAX_ROUTES = 2_000_000


def _hash(value: str, label: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a SHA-256 hex digest")


def validate(report: dict) -> tuple[list[str], list[dict], dict[str, float]]:
    if report.get("schema") != SCHEMA or report.get("benchmark_policy") != POLICY:
        raise ValueError("only the anti-contamination public-development protocol is eligible")
    if report.get("split") != "public_development" or report.get("promotion_eligible") is not False:
        raise ValueError("private or promotion-eligible scores cannot guide mixture search")
    if report.get("training_overlap_check") != "passed":
        raise ValueError("training-overlap check is absent or failed")
    for field in ("item_bank_sha256", "prompt_pack_sha256", "decode_profile_sha256"):
        _hash(report.get(field, ""), field)
    models = report.get("models")
    if not isinstance(models, list) or not 2 <= len(models) <= 24:
        raise ValueError("mixture search needs 2 to 24 measured models")
    tasks = report.get("tasks")
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 12 or len(set(tasks)) != len(tasks):
        raise ValueError("tasks must be a nonempty unique list of at most 12 names")
    if any(not isinstance(task, str) or not task for task in tasks):
        raise ValueError("invalid task name")
    weights = report.get("task_weights") or {task: 1 / len(tasks) for task in tasks}
    if set(weights) != set(tasks) or any(not isinstance(value, (int, float)) or not math.isfinite(value)
                                          or value <= 0 for value in weights.values()):
        raise ValueError("task weights must be finite and positive for every task")
    total = sum(weights.values())
    weights = {task: float(weights[task] / total) for task in tasks}
    seen = set()
    common_items: dict[str, tuple[str, int]] = {}
    for row in models:
        if not isinstance(row, dict):
            raise ValueError("each model needs a measurement record")
        name = row.get("model")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("model IDs must be nonempty and unique")
        seen.add(name)
        _hash(row.get("weights_sha256", ""), f"{name} weights_sha256")
        if row.get("access") != "local" or row.get("energy_scope") != "gpu_board":
            raise ValueError("all compared models need local access and GPU-board energy")
        vram = row.get("resident_vram_gb")
        if not isinstance(vram, (int, float)) or not math.isfinite(vram) or vram < 0:
            raise ValueError(f"{name} needs measured resident VRAM")
        measurements = row.get("tasks")
        if not isinstance(measurements, dict) or set(measurements) != set(tasks):
            raise ValueError(f"{name} needs exactly the same measured tasks")
        for task, item in measurements.items():
            if not isinstance(item, dict):
                raise ValueError(f"{name}/{task} needs a measurement record")
            if not isinstance(item.get("n"), int) or item["n"] < 30:
                raise ValueError(f"{name}/{task} needs at least 30 public development items")
            _hash(item.get("item_ids_sha256", ""), f"{name}/{task} item_ids_sha256")
            identity = (item["item_ids_sha256"], item["n"])
            if task in common_items and identity != common_items[task]:
                raise ValueError(f"{name}/{task} uses different prompts than another model")
            common_items[task] = identity
            item_scores = item.get("item_scores")
            if (not isinstance(item_scores, list) or len(item_scores) != item["n"]
                    or any(not isinstance(value, (int, float)) or not math.isfinite(value)
                           or not 0 <= value <= 1 for value in item_scores)):
                raise ValueError(f"{name}/{task} needs one finite public item score per prompt")
            for field in ("quality", "latency_s", "gpu_board_wh_per_answer", "decode_tps"):
                value = item.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"{name}/{task} lacks finite {field}")
            if (not 0 <= item["quality"] <= 1 or item["latency_s"] <= 0
                    or item["gpu_board_wh_per_answer"] < 0 or item["decode_tps"] <= 0):
                raise ValueError(f"{name}/{task} has invalid score, latency, GPU-board energy or decode t/s")
            if abs(sum(item_scores) / item["n"] - item["quality"]) > 1e-6:
                raise ValueError(f"{name}/{task} quality differs from its per-item scores")
    return tasks, models, weights


def _metrics(route: dict[str, str], rows: dict[str, dict], weights: dict[str, float]) -> dict:
    quality = latency = energy = 0.0
    for task, model in route.items():
        item = rows[model]["tasks"][task]
        weight = weights[task]
        quality += weight * item["quality"]
        latency += weight * item["latency_s"]
        energy += weight * item["gpu_board_wh_per_answer"]
    members = sorted(set(route.values()))
    return {"route": route, "models": members, "quality": round(quality, 6),
            "mean_latency_s": round(latency, 6),
            "decode_tps_by_task": {task: rows[model]["tasks"][task]["decode_tps"]
                                   for task, model in route.items()},
            "gpu_board_wh_per_answer": round(energy, 6),
            "resident_vram_gb": round(sum(rows[name]["resident_vram_gb"] for name in members), 6)}


def search(report: dict, *, max_models: int = 3, vram_budget_gb: float = 80,
           max_mean_latency_s: float | None = None, max_gpu_board_wh: float | None = None) -> dict:
    tasks, models, weights = validate(report)
    if not 1 <= max_models <= 4 or vram_budget_gb <= 0:
        raise ValueError("invalid model count or VRAM budget")
    max_models = min(max_models, len(models))
    if max_mean_latency_s is not None and max_mean_latency_s <= 0:
        raise ValueError("latency budget must be positive")
    if max_gpu_board_wh is not None and max_gpu_board_wh < 0:
        raise ValueError("GPU-board energy budget cannot be negative")
    rows = {row["model"]: row for row in models}
    names = sorted(rows)
    count = sum(math.comb(len(names), n) * n ** len(tasks)
                for n in range(1, max_models + 1))
    if count > MAX_ROUTES:
        raise ValueError(f"search space {count} exceeds bounded limit {MAX_ROUTES}")
    solos = [_metrics(dict.fromkeys(tasks, name), rows, weights) for name in names]
    def within_limits(row):
        return (row["resident_vram_gb"] <= vram_budget_gb
                and (max_mean_latency_s is None or row["mean_latency_s"] <= max_mean_latency_s)
                and (max_gpu_board_wh is None or row["gpu_board_wh_per_answer"] <= max_gpu_board_wh))

    eligible_solos = [row for row in solos if within_limits(row)]
    best_solo = (max(eligible_solos, key=lambda row: (row["quality"], -row["mean_latency_s"],
                                                   -row["gpu_board_wh_per_answer"]))
                 if eligible_solos else None)
    random_quality = sum(row["quality"] for row in solos) / len(solos)
    random_latency = sum(row["mean_latency_s"] for row in solos) / len(solos)
    random_energy = sum(row["gpu_board_wh_per_answer"] for row in solos) / len(solos)
    oracle = 0.0
    for task in tasks:
        n = rows[names[0]]["tasks"][task]["n"]
        oracle += weights[task] * sum(
            max(rows[name]["tasks"][task]["item_scores"][index] for name in names)
            for index in range(n)) / n
    top: list[dict] = []
    feasible = 0
    lowest_energy_above_solo = None
    for size in range(1, max_models + 1):
        for members in itertools.combinations(names, size):
            resident = sum(rows[name]["resident_vram_gb"] for name in members)
            if resident > vram_budget_gb:
                continue
            for assignment in itertools.product(members, repeat=len(tasks)):
                if len(set(assignment)) != size:
                    continue
                route = dict(zip(tasks, assignment))
                result = _metrics(route, rows, weights)
                if not within_limits(result):
                    continue
                feasible += 1
                if (best_solo is not None and result["quality"] >= best_solo["quality"]
                        and (lowest_energy_above_solo is None
                             or (result["gpu_board_wh_per_answer"], result["mean_latency_s"])
                             < (lowest_energy_above_solo["gpu_board_wh_per_answer"],
                                lowest_energy_above_solo["mean_latency_s"]))):
                    lowest_energy_above_solo = result
                top.append(result)
                top.sort(key=lambda row: (-row["quality"], row["gpu_board_wh_per_answer"],
                                          row["mean_latency_s"], row["resident_vram_gb"], row["models"]))
                del top[10:]
    return {"schema": "toddler-mom-search-result/v1",
            "status": "research_candidate_only" if top else "no_feasible_route",
            "source_item_bank_sha256": report["item_bank_sha256"],
            "protocol": report["benchmark_policy"], "models_measured": len(models),
            "routes_considered_upper_bound": count, "feasible_routes": feasible,
            "best_single": best_solo, "random_router_expected_quality": round(random_quality, 6),
            "oracle_router_ceiling": round(oracle, 6),
            "oracle_gain_over_best_single": round(oracle - best_solo["quality"], 6)
            if best_solo else None,
            "random_router_expected": {"quality": round(random_quality, 6),
                                       "mean_latency_s": round(random_latency, 6),
                                       "gpu_board_wh_per_answer": round(random_energy, 6)},
            "lowest_energy_at_least_best_single_quality": lowest_energy_above_solo,
            "candidates": top, "energy_scope": "gpu_board",
            "warning": "Public development selection only; use fresh sealed items and measured end-to-end runtime before promotion."}


def search_file(path: Path, **budgets) -> dict:
    data = path.read_bytes()
    result = search(json.loads(data), **budgets)
    result["source_sha256"] = hashlib.sha256(data).hexdigest()
    result["source"] = str(path.resolve())
    return result
