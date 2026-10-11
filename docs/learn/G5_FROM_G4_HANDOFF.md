# G5 handoff from the verified G4 navigation survivor

Status 11 October 2026: **a G5 public prototype exists; G5 is not promoted.** G4-search survived for
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

## First public G5 cognitive prototype

Teacher exports a fixed, deterministically keyed 27-item development lesson:
three examples each of addition, subtraction and multiplication in English,
Dutch and written Cantonese. The export requires committed, clean Teacher
source. Toddler independently recomputes every answer key, verifies the
source IDs and learns a rule only when verified feedback rules out the other
hypotheses. Its old G4 cognitive head remains the fallback. The G4 navigation
routers are loaded by hash from the five frozen parents and never rewritten.
`toddler/g5_system.py` composes the resulting cognitive head with one inherited
navigation parent at runtime, rejecting changed artifacts or parent hashes.

```bash
PYTHONPATH=/path/to/teacher python3 -m teacher.g5_feedback \
  --out /private/g5-public/teacher-lesson.json
PYTHONPATH=/path/to/toddler python3 scripts/g5_public_candidate.py \
  --teacher-curriculum /private/g5-public/teacher-lesson.json \
  --out /private/g5-public/candidate-001
```

The first local prototype, created before the new source was committed,
measured 0.2685 for the unchanged cognitive head versus 1.0000 after feedback
on 108 new **public, same-template** arithmetic questions. The inherited
public G4 retention exercise remained 0.9681 in both variants. Its old
`code_version` ignored untracked source and incorrectly said `dirty=false`;
that artifact is preserved as `INVALID-untracked-source` and cannot be used
for promotion. The corrected lineage helper counts untracked code, and the
candidate builder now refuses dirty source. A clean replacement was created
from committed code and Teacher lesson SHA-256
`1b2b5434fa0bd8639f73a2129d8efaaa20b01f4c85df65d91f53a37a7d9b18d4`;
all five composed runtimes loaded and produced the same CartPole action as
their exact frozen G4 parent. The public aggregate is
[`reports/g5_public_feedback_20261011.json`](../../reports/g5_public_feedback_20261011.json).

These 108 items test nine exact syntactic forms with new operands. They do not
test general mathematical reasoning, Cantonese comprehension, the other 17
cognitive areas, real-world action or independent holdout transfer. Teacher's
training file and the public selection set are spent for development. G5
remains an unpromoted candidate until the separate evaluator freezes and runs
new, independently reviewed tests with matched G4 baselines and navigation
retention checks.

## One-use transfer diagnostic

`scripts/g5_private_transfer_diagnostic.py` accepts an evaluator-owned,
owner-only question bank outside Git. The bank names the exact frozen candidate
manifest hash; every answer is recomputed from the declared integer operation
and operands, and exact public training forms are rejected. It compares G4 and
G5 on the same questions after all four option rotations, records paired
outcomes outside Git, and prints only aggregate counts. Each of the nine
language-operation cells needs at least eight distinct paraphrased items.
This is a **post-freeze diagnostic**, not an independent generation promotion
trial. Its question family must never be used to tune this candidate or to
claim general reasoning, language comprehension, or navigation retention.

```bash
PYTHONPATH=. python3 scripts/g5_private_transfer_diagnostic.py \
  --bank /private/g5-diagnostic/bank.json \
  --manifest /private/g5-public/candidate-001/manifest.json \
  --out /private/g5-diagnostic/result.json
```

The 11 October diagnostic was preregistered against the already frozen first
candidate, then run once on 108 paraphrased arithmetic questions. G4 and G5
both answered 33/108 correctly under four option rotations; the paired
difference was exactly zero. The first cognitive head therefore has **no
observed paraphrase-transfer gain**. The private question bank and individual
outcomes are not training material. The safe aggregate and commitments are in
[`reports/g5_transfer_diagnostic_20261011.json`](../../reports/g5_transfer_diagnostic_20261011.json).
New public training can teach more varied language, but any resulting G5
candidate must be evaluated on a newly sealed family; this bank is spent.

## Second public candidate: operation learned across paraphrases

Teacher's second lesson supplies 81 verified public training items and 108
public check items. The new G5 semantic expert masks the operand values,
infers each operation from the verified numerical answer, and learns character
patterns across English, Dutch and written Cantonese questions. Two classifier
architectures and two confidence floors are compared on the same public check.
The exact-form rule head remains available, followed by the semantic expert
and finally the unchanged G4 fallback. Verified public feedback can be added
and the expert refitted; no private item may enter that memory.

```bash
PYTHONPATH=. python3 scripts/g5_semantic_candidate.py \
  --old-lesson /private/g5-public/teacher-lesson.json \
  --semantic-lesson /private/g5-public/teacher-semantic-lesson.json \
  --out /private/g5-public/semantic-candidate-001
```

The code requires clean committed Toddler source and owner-only Teacher files,
recomputes every answer key, checks source IDs and train/check balance, and
binds five new child IDs to the same five frozen G4 navigation parents. The
earlier private transfer bank is excluded. The 108 public check questions were
used to select the architecture; their score is only a development result.
Before any G5 promotion attempt, an evaluator must preregister and independently
review a **new** cognitive bank, matched G4 comparisons, new navigation seeds,
retention margins and multiple-attempt corrections. Cantonese wording needs
native-language review before a language claim.

The second candidate was built from clean commit `f45f29bb` and Teacher lesson
SHA-256 `848eea8601e57867782962558041ab5e5f4daae5000e61af602b6daf561b039f`.
All five composed runtimes loaded with the exact frozen G4 parent navigation
hashes and returned the same CartPole action as their parent on a probe.
Public architecture selection gave 108/108 on the fourth Teacher pattern,
versus 36/108 for both G4 and the first G5 candidate. On a **new**, sealed,
post-freeze diagnostic with 108 different paraphrases, G5 scored 67/108 and
G4 scored 32/108. The paired result had 35 G5-only correct items and no
G4-only correct items (two-sided exact sign-test p = 5.82e-11). This is the
first observed arithmetic paraphrase-transfer gain, with the strongest cells
in subtraction and weak multiplication cells. It is not a G5 promotion:
the diagnostic lacks independent Cantonese language review, broad cognitive
retention, new navigation seeds and two preregistered confirmation families.
See [`reports/g5_semantic_transfer_20261011.json`](../../reports/g5_semantic_transfer_20261011.json)
for prompt-free aggregates and file commitments. Both diagnostic banks are
spent and must remain excluded from future training and selection.

## Prospective arithmetic confirmation gate

`scripts/g5_prospective_gate.py` freezes the second candidate, the five G4
parents, two new evaluator-owned banks of 72 English/Dutch arithmetic questions
each, the old cognitive fallback, and two newly committed eight-seed navigation
sets. It rejects exact prompt reuse from both Teacher lessons and both spent
transfer diagnostics. Each item is checked against a deterministic answer
function; each trial scores G4 and G5 on the same item with four option
rotations. The nine navigation tasks are run for every child and its exact
G4 parent on matched secret seeds. An interruption can resume only the same
frozen attempt.

Before either result is seen, the gate requires on **each** 72-item trial:
at least 44 correct G5 answers, at least 12 net paired gains, at most two
G4-only losses, no negative language-operation cell and a one-sided exact
paired sign-test p &lt; 0.005. Navigation must match its frozen parent on every
measured seed. The non-arithmetic G4 fallback model parameters must be exactly
equal; the new expert is gated to arithmetic inputs. The two trials are
separate confirmation families, not two views of the earlier diagnostic.

The assessment deliberately returns `evidence_passed_pending_semantic_review`
or `failed_hold_no_promotion`; it does **not** append a `survived` lineage
event. The question wording still needs independent semantic review. A full
G5 generation claim additionally needs explicit scope registration, source
bundle import into the lineage and checks for other requested capabilities;
Cantonese and physical interaction are outside this arithmetic gate.

```bash
PYTHONPATH=. python3 scripts/g5_prospective_gate.py prepare \
  --manifest /private/g5-public/semantic-candidate-001/manifest.json \
  --reference /private/g5-public/first-candidate/cognitive.joblib \
  --bank /private/g5-confirm/set-a.json --bank /private/g5-confirm/set-b.json \
  --exclude /private/g5-public/teacher-lesson.json \
  --exclude /private/g5-public/teacher-semantic-lesson.json \
  --exclude /private/g5-diagnostic/first-bank.json \
  --exclude /private/g5-diagnostic/second-bank.json \
  --out /private/g5-confirm/audit-001
PYTHONPATH=. python3 scripts/g5_prospective_gate.py run \
  --protocol /private/g5-confirm/audit-001/protocol.json --index 0
PYTHONPATH=. python3 scripts/g5_prospective_gate.py run \
  --protocol /private/g5-confirm/audit-001/protocol.json --index 1
PYTHONPATH=. python3 scripts/g5_prospective_gate.py assess \
  --protocol /private/g5-confirm/audit-001/protocol.json
```

### 11 October confirmation result

The first clean prospective attempt used a second, disjoint pair of
owner-only banks. Each bank contained 72 new English/Dutch arithmetic items;
four earlier Teacher or diagnostic files were explicitly excluded. The
protocol, candidate manifest, frozen parent contract, evaluator commit and
bank hashes are recorded in
[`reports/g5_prospective_gate_20261011.json`](../../reports/g5_prospective_gate_20261011.json).

The two confirmation families were deliberately conjunctive. Trial A scored
G4 18/72 and G5 38/72, with 20 G5-only items and no G4-only items. Trial B
scored G4 19/72 and G5 47/72, with 28 G5-only items and no G4-only items.
Both trials retained all measured navigation actions exactly and had no
negative language-operation cell. Trial B met every threshold; Trial A did
not meet the preregistered 44/72 G5 floor. The resulting status is
**`failed_hold_no_promotion`**. No lineage `survived` event was written, and
the candidate must not be presented as a promoted G5. The failed confirmation
family is spent; it cannot be used to tune the candidate. A future attempt
needs a fresh, independently reviewed bank and must retain the same frozen
parent contract.

## G5.1 public curriculum candidate

Teacher's G5.1 curriculum adds 144 new public development items: 72 train and
72 check items across balanced English/Dutch addition, subtraction and
multiplication cells. The questions use story and comparative wording rather
than the earlier direct forms. Dutch text is explicitly marked
`pending_native_review`; the candidate cannot be treated as language evidence
until a native reviewer signs the wording and answer keys.

The clean G5.1 candidate selected a hashing character SGD expert at confidence
floor 0.5 on the public check. It scored 54/72 (0.750000), while the unchanged
G4 baseline and previous candidate each scored 19/72 (0.263889). Public
retention stayed 0.968085 for both G4 and G5.1. Cell aggregates were 4/12,
12/12 and 12/12 for English addition, subtraction and multiplication, and
2/12, 12/12 and 12/12 for the corresponding Dutch cells. The result is
development evidence only: it selected an architecture, is not a private
promotion score, and does not establish Cantonese, broad cognition, software,
physical action or new navigation retention. The prompt-free aggregate and
artifact commitments are in
[`reports/g5_1_public_candidate_20261011.json`](../../reports/g5_1_public_candidate_20261011.json).
