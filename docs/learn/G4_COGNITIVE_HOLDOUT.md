# G4 cognitive holdout pilot

The current G4 cognitive head is a TF-IDF answer ranker trained on a small
public fact catalog and generated additions. Its public development score
0.9681 reuses the fact question templates; it is not evidence of broad
language, reasoning or Montessori competence.

An evaluator-owned, owner-readable JSON bank outside Git now contains 54 new
questions: three in each of the 12 knowledge areas and six Montessori-inspired
areas. The bank SHA-256 is
`5a4baa6815f24eaa49f543bc52734239c4e38ec488864194d291985771c0b543`.
The scorer rejects public-training question duplicates, checks answer keys and
area coverage, and tests every item under four answer-order rotations. The
bank and question-level predictions are never published while held out.

The first diagnostic gave **22/54 = 0.4074 strict accuracy**. Arithmetic,
written Cantonese, history, geography, trivia, IQ reasoning and Montessori
sensorial each scored 0/3. Each area has only three examples, and the bank's
elementary answer keys were checked locally, not by an independent human
reviewer. These are research signals, not precise per-area ability estimates
or a promotion score. The poor transfer shows why new question forms and
verified content are needed before cognitive skills enter the generation gate.

The private JSON uses schema `toddler-g4-cognitive-holdout/v1` and rows with
`area`, `question`, four `options`, `answer` and `key_source`. A separate
evaluator can score a future, independently reviewed bank with:

```bash
PYTHONPATH=. python3 scripts/g4_cognitive_holdout.py \
  --bank /private/questions.json --model /private/cognitive.joblib \
  --out /private/aggregate.json
```

Any model revised after seeing this diagnostic needs a **new** unseen bank;
the present one must never become training or development material. A future
cognitive promotion protocol must specify enough independently checked items
per area, matched baselines and a frozen pass rule before evaluation.
