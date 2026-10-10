"""Estimate logical and deduplicated storage for the 10,000-slot atlas."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ATLAS = Path(__file__).resolve().parents[1] / "knowledge" / "atlas.json"
MOUNTS = (Path("/media/knight2/eds1"), Path("/media/knight2/EDS2"),
          Path("/media/knight2/claude-data"))
GB = 10**9

# Topic bundles include deduplicated papers, web snapshots, videos, repositories
# and shared datasets. Unique skill data include instructions, exercise assets,
# submissions and provenance. Index is a separate graph/vector/codegraph budget.
SCENARIOS = {
    "lean": {"topic_bundle_gb": 2, "unique_skill_mb": 20, "index_gb": 20},
    "balanced": {"topic_bundle_gb": 8, "unique_skill_mb": 100, "index_gb": 50},
    "rich": {"topic_bundle_gb": 20, "unique_skill_mb": 500, "index_gb": 200},
}


def estimate(slots: int, topics: int, *, topic_bundle_gb: float,
             unique_skill_mb: float, index_gb: float, working_headroom: float = .20) -> dict:
    if slots < 1 or topics < 1 or slots < topics or not 0 <= working_headroom < 1:
        raise ValueError("invalid atlas size or headroom")
    if min(topic_bundle_gb, unique_skill_mb, index_gb) < 0:
        raise ValueError("storage units cannot be negative")
    unique_gb = unique_skill_mb / 1000
    logical_per_slot_gb = topic_bundle_gb + unique_gb + index_gb / slots
    logical_total_gb = logical_per_slot_gb * slots
    primary_gb = topics * topic_bundle_gb + slots * unique_gb + index_gb
    return {"logical_per_slot_gb": round(logical_per_slot_gb, 3),
            "logical_total_tb": round(logical_total_gb / 1000, 3),
            "physical_amortized_per_slot_mb": round(primary_gb * 1000 / slots, 1),
            "primary_tb": round(primary_gb / 1000, 3),
            "with_working_headroom_tb": round(primary_gb / (1 - working_headroom) / 1000, 3),
            "two_copies_tb": round(2 * primary_gb / 1000, 3)}


def report(atlas: Path = ATLAS, mounts: tuple[Path, ...] = MOUNTS) -> dict:
    metadata = json.loads(atlas.read_text())
    slots, topics = metadata["slots"], metadata["topics"]
    disks = {}
    for mount in mounts:
        if mount.is_mount():
            usage = shutil.disk_usage(mount)
            disks[str(mount)] = {"free_gb": round(usage.free / GB, 1),
                                 "usable_after_200gb_reserve_gb": round(max(0, usage.free / GB - 200), 1)}
    return {"schema": "toddler-capability-storage-estimate/v1",
            "basis": {"slots": slots, "topics": topics, "units": "decimal GB/TB",
                      "working_headroom": .20, "ssd_reserve_gb_each": 200},
            "scenarios": {name: {**assumptions, **estimate(slots, topics, **assumptions)}
                          for name, assumptions in SCENARIOS.items()},
            "disks": disks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--atlas", type=Path, default=ATLAS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = json.dumps(report(args.atlas), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")
