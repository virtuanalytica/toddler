"""Paired, component-level development attribution without causal overclaim.

Inputs are task-level observations from a randomized, same-profile ablation.
This analysis estimates conditional effects; it never promotes a model.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

METRICS = ("accepted", "latency_s", "gpu_board_wh", "system_kwh")
COMPONENTS = ("pulse", "knitweb", "lightrag", "gitnexus", "codegraph",
              "jev", "clm", "local_router", "dspy_optimizer")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _interval(diffs: list[float], confidence: float, seed: int = 1729) -> list[float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(_mean([diffs[rng.randrange(n)] for _ in range(n)]) for _ in range(2000))
    lower = means[max(0, int((1 - confidence) / 2 * len(means)))]
    upper = means[min(len(means) - 1, int((1 + confidence) / 2 * len(means)))]
    return [round(lower, 6), round(upper, 6)]


def analyze(data: dict) -> dict:
    if data.get("schema") != "toddler-component-arms/v1":
        raise ValueError("unsupported attribution schema")
    if data.get("split") not in {"public_development", "private_confirmatory"}:
        raise ValueError("invalid split")
    profile = data.get("shared_profile_sha256")
    if not isinstance(profile, str) or len(profile) != 64 or any(c not in "0123456789abcdef" for c in profile):
        raise ValueError("shared model/prompt/decode/hardware profile hash required")
    randomization = data.get("randomization_sha256")
    if (not isinstance(randomization, str) or len(randomization) != 64
            or any(c not in "0123456789abcdef" for c in randomization)):
        raise ValueError("preregistered randomized run-order hash required")
    components = data.get("applicable_components")
    if (not isinstance(components, list) or not components or len(set(components)) != len(components)
            or any(c not in COMPONENTS for c in components)):
        raise ValueError("invalid component set")
    arms = data.get("arms")
    expected = {"baseline", "full"} | {f"plus/{c}" for c in components} | {f"minus/{c}" for c in components}
    if not isinstance(arms, dict) or set(arms) != expected:
        raise ValueError("missing or unexpected ablation arm")
    common_ids = None
    values = {}
    for arm, rows in arms.items():
        if not isinstance(rows, list) or len(rows) < 30:
            raise ValueError(f"{arm} needs at least 30 paired tasks")
        by_id = {}
        for row in rows:
            if not isinstance(row, dict) or row.get("profile_sha256") != profile:
                raise ValueError(f"{arm} differs from shared model/prompt/decode/hardware profile")
            item_id = row.get("item_id")
            if not isinstance(item_id, str) or not item_id or item_id in by_id:
                raise ValueError(f"{arm} has missing or duplicate item ID")
            if type(row.get("accepted")) is not bool:
                raise ValueError(f"{arm}/{item_id} lacks independent acceptance")
            for metric in METRICS[1:]:
                value = row.get(metric)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"{arm}/{item_id} lacks valid {metric}")
            by_id[item_id] = {**row, "accepted": float(row["accepted"])}
        if common_ids is None:
            common_ids = set(by_id)
        elif set(by_id) != common_ids:
            raise ValueError(f"{arm} does not use identical task IDs")
        values[arm] = by_id
    ids = sorted(common_ids)
    confidence = 1 - 0.05 / len(components)  # conservative simultaneous CI

    def paired(left: str, right: str) -> dict:
        result = {}
        for metric in METRICS:
            diffs = [values[left][key][metric] - values[right][key][metric] for key in ids]
            result[metric] = {"mean_delta": round(_mean(diffs), 6),
                              "simultaneous_ci": _interval(diffs, confidence),
                              "better_direction": "positive" if metric == "accepted" else "negative"}
        return result

    standalone = {c: paired(f"plus/{c}", "baseline") for c in components}
    conditional = {c: paired("full", f"minus/{c}") for c in components}
    full_gain = paired("full", "baseline")
    interaction = {metric: round(full_gain[metric]["mean_delta"]
                                 - sum(standalone[c][metric]["mean_delta"] for c in components), 6)
                   for metric in METRICS}
    return {"schema": "toddler-component-attribution/v1", "status": "analysis_only",
            "promotion_eligible": False, "split": data["split"], "paired_items": len(ids),
            "randomization_sha256": randomization,
            "components": components, "simultaneous_confidence": round(confidence, 6),
            "full_vs_baseline": full_gain, "standalone_effects": standalone,
            "conditional_effects": conditional, "unexplained_interaction": interaction,
            "interpretation": "Conditional associations under randomized identical-arm tasks; interactions and component dependencies prevent simple percentage attribution."}


def analyze_file(path: Path) -> dict:
    content = path.read_bytes()
    result = analyze(json.loads(content))
    result["source_sha256"] = hashlib.sha256(content).hexdigest()
    return result
