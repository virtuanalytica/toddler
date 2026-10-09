# G4 oracle cohort and private promotion protocol

Teacher trains one child for each official `G3-recombined/t5001`–`t5005` parent.
Each inherits that parent's complete task router. An oracle-taught specialist
may replace only `unlockpickup`, under one frozen public-development rule:
its mean normalised score on seeds 50,000–50,049 must exceed the matched G3
policy by at least 0.05. The new expert changes no ancestor weights or other
routes. Public scores select the route before the private protocol is frozen;
they are not promotion evidence. PPO is outside this nightly recipe.

The independent evaluator validates every artifact hash and compares each
inherited expert tensor with the official G3 parent. It creates two new
30-seed sets in the secret band, stores the raw sets under the evaluator's
private directory, and publishes only their commitments. The same frozen
weights, nine-task battery, sampled action mode and matched G3 control are used
on both sets. G1, G2 and G3 are re-evaluated on those sets. Every mastered
ancestor task retains the existing regression floor. The
`toddler.learn.ancestor_gate` requires five distinct children, both trials to
pass, p < 0.05 and probability of improvement >= 0.75 against the strongest
ancestor and the G3 control. No result can be promoted from a public score.

After Teacher produces a complete manifest, the evaluator uses:

```bash
export PYTHONPATH=.
python3 scripts/g4_private_trial.py prepare \
  --manifest /private/teacher-run/manifest.json --out /private/g4-trial
python3 scripts/g4_private_trial.py run --protocol /private/g4-trial/protocol.json --index 0
python3 scripts/g4_private_trial.py run --protocol /private/g4-trial/protocol.json --index 1
python3 scripts/g4_private_trial.py assess --protocol /private/g4-trial/protocol.json
```

Raw trial rows stay outside Git. `promote` imports weights and writes the
hash-chained G4 lineage event only when `assess` passes both trials. The public
report contains commitments, protocol/trial hashes, aggregate scores and
regression verdicts, but no hidden seeds or per-child private rows before the
scheduled reveal. A failed or incomplete run stays a candidate. The cognitive
head has a separate independent assessment track because its current public
question templates overlap training and cannot justify generation promotion.
