#!/usr/bin/env python3
"""Convert verified local-mixture evidence into Preschooler train/development JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from toddler.learn.preschooler_data import prepare_files

DEFAULT_ROSTER = Path(__file__).resolve().parents[1] / "docs/learn/preschooler-teacher-roster.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--protected-commitment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-teacher-models", type=int, default=10,
                        help="Distinct local teacher IDs required across this export (pilot: 2)")
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER,
                        help="Pinned, rights-approved teacher roster")
    args = parser.parse_args()
    if args.min_teacher_models < 1:
        parser.error("--min-teacher-models must be positive")
    manifest = prepare_files(args.evidence, args.protected_commitment, args.out,
                             min_teacher_models=args.min_teacher_models,
                             roster_path=args.roster, require_mixture=True)
    print(f"Admitted {manifest['rows']} verified local examples; manifest: {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
