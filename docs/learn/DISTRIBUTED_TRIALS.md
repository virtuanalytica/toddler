# Distributed Toddler trials

The public PR is a **method and evidence record**. Training can happen on any
machine. The contributor discloses every model and interaction that influenced
the candidate, including privately hosted models by exact ID and revision.
The independent evaluator controls the promotion holdout. A PR with only a
contributor-run benchmark can be merged as a candidate; it cannot mark the
generation `survived`.

Place these three files under `submissions/<github-user>/<candidate>/`:

- `protocol.json`: the frozen plan, with `seed_set_id`, task battery, control,
  budgets, evaluation mode, parent hashes and promotion rule. If the set is
  private, include only the set ID and commitment.
- `report.json`: the contributor's measured result, including `seed_set_id`
  and the exact `protocol_sha256`. Include score units, per-task aggregate,
  duration and hardware. Record NULL and failed runs honestly.
- `submission.json`: the recipe, model and interaction disclosure plus paths
  and SHA-256 hashes for every protocol/report pair.

Minimum `submission.json` structure (replace all example values):

```json
{
  "schema": "toddler-contribution/v1",
  "contributor": {"github": "alice", "name": "Alice Example"},
  "candidate": {
    "generation": "G4-alice", "id": "t4001",
    "weights_sha256": "<64 lowercase hex characters>",
    "parent_refs": ["G3-recombined/t5001"]
  },
  "recipe": {
    "summary": "Curriculum and frozen expert changes",
    "commands": ["python3 scripts/train_example.py --config config.json"],
    "environment": {"python": "3.12", "torch": "2.6.0", "minigrid": "3.1.0"},
    "data_sources": [{"id": "MiniGrid", "provenance": "minigrid 3.1.0", "visibility": "public"}],
    "models_used": [{
      "provider": "example-provider", "model_id": "exact-model-id",
      "revision": "exact-version-or-weight-hash", "role": "teacher",
      "usage": "proposed training curriculum", "visibility": "private"
    }],
    "interactions": [{"kind": "human feedback", "count": 20,
                      "log_sha256": "<64 lowercase hex characters>"}]
  },
  "trials": [{
    "kind": "contributor_public", "seed_set_id": "public-set-a",
    "matched_control": "G3-recombined/t5001", "eval_mode": "sample",
    "protocol": "protocol.json", "protocol_sha256": "<64 lowercase hex characters>",
    "report": "report.json", "report_sha256": "<64 lowercase hex characters>"
  }]
}
```

Use `"models_used": []` together with `"no_external_models": true` if no
outside models influenced the work. Interactions can be an empty list. A
private model can be identified without publishing its weights, credentials or
private training data. If its terms forbid disclosing the exact identity or
using its output for training, do not submit that trial as reproducible.
Artifacts must be JSON within the submission directory; the validator
rejects path escapes, raw secret fields, changed hashes and reused
contributor seed-set IDs.

## Independent ancestor audit

The evaluator selects surviving ancestors by walking the real lineage from
the candidate's parent refs. Each trial runs the candidate, a matched control
and every surviving ancestor on the **same** hidden seed set. A predecessor
with no policy for a newly introduced task receives zero for that task only in
the whole-battery comparison. Every task an ancestor mastered retains a
separate regression floor. Five distinct matched children and two independent
seed sets are required; the candidate weights, task battery and frozen plan
must be identical in both trials.

`toddler.learn.ancestor_gate` implements this decision. The evaluator can
score its private JSON result files locally with:

```bash
PYTHONPATH=. python3 scripts/assess_ancestor_trials.py \
  --root /path/to/official-generations \
  --parents G3-recombined/t5001 \
  --trial /private/a.json /private/b.json
```

The script prints only aggregate decisions; it never prints the raw private
rows. It expects the official evaluator to verify the model files and their
weights against the frozen plan first. The JSON files supply `seed_set_id`,
`frozen_plan_sha256`, `child_ids`, `candidate_weights`, and dictionaries
`candidate`, `control`, `ancestors`; each task maps to one score per matched
child. These files stay outside the Git repository until the commitments can
be revealed under their preregistered schedule.

G3's originator-signed promotion remains in the archive as the pivot. Later
promotion uses repeatable audit trials and a hash-chained lineage event rather
than another person signing each deployment. The trial author, model IDs,
recipe, code commit, weight hashes, seed commitments, aggregate outcomes and
remaining failures remain auditable.
