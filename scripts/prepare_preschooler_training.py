#!/usr/bin/env python3
"""Convert verified local-mixture evidence into Preschooler train/development JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from toddler.learn.preschooler_data import prepare_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--protected-commitment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_files(args.evidence, args.protected_commitment, args.out)
    print(f"Admitted {manifest['rows']} verified local examples; manifest: {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
