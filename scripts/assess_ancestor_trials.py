"""Assess preregistered private descendant trials without publishing their rows.

Input files stay outside Git; stdout contains the decision and aggregate
statistics only. The official evaluator must independently create the hidden
seed sets, run the frozen candidate, controls and all surviving ancestors, and
verify every weight hash before calling this scorer.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from toddler.learn.ancestor_gate import Trial, assess_lineage_replication
from toddler.learn.lineage import Ledger


def _trial(path: Path) -> Trial:
    data = json.loads(path.read_text())
    return Trial(data["seed_set_id"], data["frozen_plan_sha256"], tuple(data["child_ids"]),
                 data["candidate_weights"],
                 {key: tuple(value) for key, value in data["candidate"].items()},
                 {key: tuple(value) for key, value in data["control"].items()},
                 {name: {key: tuple(value) for key, value in scores.items()}
                  for name, scores in data["ancestors"].items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--parents", required=True, nargs="+")
    parser.add_argument("--trial", required=True, nargs="+", type=Path)
    args = parser.parse_args()
    ledger = Ledger(args.root)
    if ledger.verify() != (True, None):
        parser.error("official lineage hash chain is invalid")
    decision = assess_lineage_replication(ledger, tuple(args.parents), tuple(_trial(p) for p in args.trial))
    summary = {"promote": decision.promote, "required_ancestors": decision.required_ancestors,
               "trials": [{"passed": row.passed, "strongest_ancestor": row.strongest_ancestor,
                           "vs_ancestor": asdict(row.promotion_vs_ancestor),
                           "vs_control": asdict(row.promotion_vs_control),
                           "retention_floors": row.retention_floors,
                           "regressions": row.regressions}
                          for row in decision.trials]}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
