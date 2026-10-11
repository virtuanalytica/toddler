# G5 handoff from the verified G4 navigation survivor

Status 11 October 2026: **G5 is not running or promoted.** G4-search survived for
navigation on two sealed, matched 30-seed trials. The current Teacher
`g4_nightly.py` still loads `G3-recombined/t5001..t5005` and emits
`G4-candidate/t6001..t6005`; its separate `teacher-g4.timer` is not installed
in the active user systemd instance. Repeating that service cannot create an
auditable G5. Do not relabel its children or reuse the spent G4 private family.

## Why the next generation needs a new target

The five G4 children have private aggregate navigation scores near or above 1.0
on all nine tasks, including `unlockpickup` (IQM 0.999). A 54-item independent
cognitive diagnostic for the current TF-IDF head scored 22/54. It was a small,
partly locally keyed pilot, and its questions have now influenced development;
none may be used as G5 final-test items. G5 should test whether a new cognitive
capability can be added while preserving the verified G4 navigation policy.
Software work and physical action remain separate task families.

## Required implementation before a G5 timer is enabled

1. **Freeze ancestry.** Load exactly the five `G4-search/t6001..t6005` parents
   from the verified lineage, check their weight hashes and `survived` verdict,
   and register five distinct G5 child IDs. The initial public parent contract
   is [`reports/g5_frozen_g4_parents_20261011.json`](../../reports/g5_frozen_g4_parents_20261011.json).
   `python3 scripts/g5_parent_contract.py --verify reports/g5_frozen_g4_parents_20261011.json`
   checks the hash-chained lineage,
   two confirmatory trial commitments, all five registry weights, nine routes
   and each expert's tensor contents. Candidate manifests must name the
   inherited module hashes and every training source. Never mutate G4 weights.
2. **Search under a fixed budget.** Compare an unchanged G4 baseline, a
   verified-answer Teacher curriculum, own-interaction or self-reward learning,
   and distinct architecture or model-routing variants. Use identical public
   development/check splits and budgets within each comparison. Record failed
   recipes and rollbacks as well as selected ones. CLM, JEV and mixture routes
   require separate ablations before claiming their contribution.
3. **Make a new cognitive test.** Independently review answer keys for new
   arithmetic, language and Montessori items across the 18 named areas. Lock
   source hashes, contamination checks, sample size, primary score, paired
   comparison, non-regression margins and multiple-attempt policy before any
   candidate sees the items. The old 54-item bank remains diagnostic history.
4. **Retain navigation.** Evaluate the nine inherited tasks on new matched
   seeds against each frozen G4 parent, with a predeclared tolerance. A
   cognitive gain cannot erase a verified navigation skill. Compare speed and
   energy under the same hardware profile when the new component uses LLMs.
5. **Promote only through a new private family.** The evaluator, not Teacher,
   owns the hidden items and seeds. It writes immutable commitments and two
   independent confirmations, validates the manifest and code version, then
   appends a `survived` or `extinct` G5 verdict to the lineage. A failure or
   interruption resumes the same frozen attempt; it never gets fresh secret
   items for free.
6. **Activate the correct service.** Replace the G3-specific G4 entrypoint with
   a G5 entrypoint only after it passes the parent, manifest, contamination and
   interruption tests. Install its own timer with benchmark-slot awareness;
   do not schedule the current G4 service as a substitute.

The current authoritative evidence is the lineage under
`/media/knight2/EDS2/toddler-generations`, the owner-only G4 audit directory,
[`G4_AUTO_AUDIT.md`](G4_AUTO_AUDIT.md), and
[`G4_COGNITIVE_HOLDOUT.md`](G4_COGNITIVE_HOLDOUT.md). Public dashboard JSON
contains aggregates only; it cannot substitute for the private promotion
record.
