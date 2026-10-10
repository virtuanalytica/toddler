#!/usr/bin/env python3
"""Render a private/customer Toddler work-pilot scorecard from metadata-only JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toddler.work_pilot import load_results, score_paired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_jsonl", type=Path)
    parser.add_argument("--candidate", choices=("toddler_teacher", "toddler_teacher_agent"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    scorecard = score_paired(load_results(args.evidence_jsonl), args.candidate)
    output = json.dumps(scorecard, indent=2) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
