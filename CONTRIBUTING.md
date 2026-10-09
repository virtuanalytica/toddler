# Contribute a stronger Toddler

Toddler accepts independent experiments through GitHub pull requests. You can
develop on your own hardware and use your own interactions, teachers, and
public or private models. Disclose **how** the candidate was made so other
people can judge the method. The code and documentation in this repository
use Apache-2.0; Wikipedia extracts in `data/corpus/` use CC BY-SA 4.0. See
[`NOTICE`](NOTICE). Do not submit model weights, datasets or outputs unless
you have the right to redistribute them.

## Submit a candidate

1. Fork the public repository and create a branch. Record the exact parent
   refs and hashes, task battery, comparison rule, budgets and seed-set
   commitment before the first evaluation. Keep private seeds and salt off
   GitHub. Training seeds and official holdout seeds must be disjoint.
2. Put `submission.json`, `protocol.json` and `report.json` in
   `submissions/<github-user>/<candidate>/`. The manifest format is described
   in [`docs/learn/DISTRIBUTED_TRIALS.md`](docs/learn/DISTRIBUTED_TRIALS.md).
   Disclose every model used to generate data, propose a curriculum, judge,
   filter, review or train the candidate: provider, exact model ID/revision,
   role, usage and whether it is public or private. The model itself need not
   be redistributed. If no outside model was used, say so explicitly. List
   interaction kinds, counts, log hashes and data provenance.
3. Include a matched G3 or later baseline and a control with the same task
   battery, decode/action mode, step budget and evaluation settings. Explain
   failed trials and method changes; a changed recipe needs a new protocol.
4. Run `python3 scripts/validate_contribution.py --manifest
   submissions/<github-user>/<candidate>/submission.json` and
   `python3 -m pytest -q`, then open a PR using the Toddler trial template.
   The manifest check verifies disclosure and report/protocol hashes. It does
   **not** certify quality or reproduce private trials.
5. The independent evaluator freezes the submitted weights and plan, draws
   fresh hidden seed sets, reruns candidate, control and every surviving
   ancestor on matched seeds, and evaluates the ancestor gate twice without
   changing weights. Public results include aggregate metrics, hashes and
   limitations, never unrevealed seed values or salt. A passing candidate may
   be promoted automatically with a verifiable lineage event. G3 was the last
   originator-signed pivot; later Toddler trials do not require that signature.

Merging a PR publishes the method and evidence, **not** an official `survived`
verdict. A contributor with repository write/merge access may merge their own
PR after checks and review. Other contributors use a fork PR and a maintainer
merges it. The independent promotion result is recorded separately in the
generation registry and dashboard.

For public forks, do not put API keys, private prompts, personal information,
unreleased model weights, raw private test items, hidden seeds or salts in the
branch, commit history, PR text or CI logs. Report only commitments, counts,
content hashes and licensed aggregate outcomes where source terms permit.

The current ancestor gate requires five matched children per arm, a one-sided
promotion comparison at p < 0.05 and probability of improvement >= 0.75
against the strongest surviving ancestor and matched control on each of two
independent hidden seed sets. If an ancestor reached 0.90 IQM on a task, the
candidate must retain at least `max(0.90, best_ancestor_IQM - 0.10)` on that
task. These thresholds must be frozen in the protocol before evaluation.
