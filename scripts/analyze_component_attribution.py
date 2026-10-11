#!/usr/bin/env python3
"""Summarize an existing same-task component-ablation record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from toddler.learn.component_attribution import analyze_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_file(args.input)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
