"""Inspect one capability's material, source candidates and codegraph connection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def lookup(capability_id: str, root: Path = ROOT) -> dict:
    ids = set((root / "knowledge" / "CAPABILITY_INDEX.txt").read_text().splitlines())
    if capability_id not in ids:
        raise KeyError(f"unknown capability ID: {capability_id}")
    folder = root / "knowledge" / "capabilities" / capability_id
    metadata = json.loads((folder / "capability.json").read_text())
    links_path = root / "knowledge" / "graph" / "capability_repository_links.jsonl"
    link = None
    with links_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["capability_id"] == capability_id:
                link = row
                break
    if link is None:
        raise ValueError("capability has no repository/codegraph entry")
    tsne = json.loads((root / "knowledge" / "graph" / "lesson_tsne.json").read_text())
    visual = next((row for row in tsne["nodes"] if row["id"] == capability_id), None)
    return {"capability": metadata, "lesson": (folder / "lesson.md").relative_to(root).as_posix()
            if (folder / "lesson.md").is_file() else None,
            "links": link, "exploratory_tsne": visual}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("capability_id")
    args = parser.parse_args()
    print(json.dumps(lookup(args.capability_id), ensure_ascii=False, indent=2))
