# Task plan: generations of toddlers

What is trained, in which order, and when a step counts as done. Each step is one branch and one reviewed pull request.

| Step | Goal | Done when | Status |
| ---- | ---- | --------- | ------ |
| A Resource governor | Use only free GPU memory, never dominate customer workloads | Unit tests for budget, fail-closed guard, timeouts, back-off; live probe skips busy GPUs | done (#7) |
| B Learning toddlers | PPO on real Gymnasium tasks with comparable scoring | CPU runs reproducible from seed and step budget; held-out evaluation seeds disjoint from training; CartPole learns (normalised > 0.2 on at least one of two seeds at 150k steps) | done (#9; return scaling #18: Acrobot and CartPole solved at 150k steps; MountainCar still 0) |
| C Toddlers teach toddlers | Show that a trained toddler helps a younger one | Real teacher beats BOTH no teacher and a random teacher (one-sided Mann-Whitney, alpha 0.05, 5 seeds, equal step budget) under a protocol registered before the run | done: KL NULL (#11); behaviour cloning passes and replicates (#12) |
| D Generations | Save, trace and promote generations | Register with weights + sha256, lineage, checkpoints, device switches and business fields; promotion needs at least 5 seeds, p < 0.05 and P(improvement) >= 0.75 | done (#15): gen-0 vs gen-1 (BC) at 150k steps, promotion refused (p 0.79); quotients gen-0 100.0, gen-1 91.6 |
| E Benchmarks | IQ / EQ / FQ per generation and per hardware configuration | Quality measured hardware-independently on the CPU; efficiency (time, energy) per hardware configuration; skipped configurations reported explicitly; frozen reference generation re-measured every run | PR #16: quality reproduced 10/10, drift 0; GPUs skipped while the guard's preflight script is missing; EQ/FQ not yet measured |
| F Literature | Keep the method grounded | Every technique in the code points to a primary source | this document and `TRAINING_LITERATURE.md` |

## Next generation status (9 October 2026)

**G3-recombined is the official surviving generation.** Its five frozen
task-expert children passed the pre-registered comparison against G2 and a
random route on two independent secret seed sets. The separate human SSH
signature was verified and the five exact candidate weights were imported
into the hash-chained lineage; see the [promotion record](learn/G3_PROMOTION_20261009.md).
The source [review packet](learn/G3_RECOMBINED_REVIEW.json) remains immutable
and says `pending_human_review` because the approval was signed separately.
The [CPU serving probe](learn/G3_RUNTIME_BENCHMARK_20261009.md) measured
near-G2 forward latency and about five times G2's saved weight size.
UnlockPickup remained unsolved. A new training trial may not tune against
either completed secret seed set.

G3 is the last promotion with a human originator signature (Deve Luse;
signature identity `knight2`). Later candidates can be developed on independent
machines and submitted by PR with complete model, data, interaction and trial
provenance; see [distributed trials](learn/DISTRIBUTED_TRIALS.md). The
[ancestor gate](../toddler/learn/ancestor_gate.py) re-evaluates every surviving
ancestor on matched hidden seeds, enforces retention of mastered tasks and
requires two unchanged-weight, independent trials against both the strongest
ancestor and a matched control. A PR merge records a contribution; a passing
independent audit records a survivor.

## G4 vervolg (9 oktober 2026)

De Teacher maakt nu elke nacht vijf oudergebonden oracle-kinderen. De eerste
bevroren cohortproef verhoogde de geheime negen-taak-IQM van circa 0,946
(G3) naar 1,042, zonder behoudsregressies, maar alleen de tweede van twee
afgeschermde sets haalde de vooraf vastgelegde p-grens. **G4 blijft kandidaat.**
Zie het [volledige auditverdict](learn/G4_ORACLE_RESULT_20261009.md).

1. Bevries een nieuwe receptuur op openbare ontwikkelkaarten. Een eerste
   onderzoeksproef met G3-sp als start voor t6001 gaf 50/50 en mean 1,0240,
   tegenover 1,0148 voor diens G3-ouder. Dit is nog geen G4-kind of blinde score.
2. Registreer vóór een nieuwe geheime toets een volledig nieuwe cohort- en
   analyseopzet, inclusief een expliciete aanpak voor opeenvolgende pogingen.
   Gebruik geen enkele seed of individuele rij uit de twee voltooide sets om
   gewichten, routegrens of statistische toets te kiezen.
3. Verbeter de cognitieve Teacher op openbare, antwoordgeverifieerde nieuwe
   vraagvormen. De aparte [54-vragenpilot](learn/G4_COGNITIVE_HOLDOUT.md)
   haalde 22/54; het oude publieke cijfer 0,9681 overschatte transfer.
   Gebruik die pilotbank niet voor training. Laat een onafhankelijke eigenaar
   een grotere nieuwe bank per gebied valideren vóór een cognitieve poort.
4. Meet na een geslaagde navigatiepoort ook volledige episodetijd en
   systeemenergie. De huidige 0,060 ms is alleen een CPU-policy-microproef.

## Doorlopende generatiezoektocht (10 oktober 2026)

De nachtelijke Teacher is nu een begrensde onderzoeksloop. Voor elk van de
vijf oudergebonden kinderen probeert hij overdracht uit G2, verder leren
vanuit de actieve G3-expert met twee demonstratiebudgetten, twee nieuwe
netwerkbreedtes (32 en 128) en PPO-vervolgtraining op eigen simulatorbeloning.
Een gekozen specialist van de vorige nacht kan
als extra startpunt meedoen. Hiermee zoekt hij zowel netwerkarchitectuur als
een betere taakgebonden mix van bestaande en nieuwe experts. Iedere ronde
wordt gemeten op openbare ontwikkelkaarten en een afzonderlijke openbare
controle. Alleen de beste kandidaat die op beide minstens 0,005 boven de
gematchte G3-ouder ligt, vervangt diens `unlockpickup`-route in een **kandidaat**.
Alle andere G3-routes en hun gewichten blijven intact. De Teacher bewaart
profielen, bronnen, seedbereiken, gewichthashes en selectie in het manifest.
De [eerste volledige zoekronde](learn/G4_SEARCH_20261010.md) koos voor vier
van vijf kinderen een nieuwe route; t6001 bleef bij G3. De private
voorbereiding heeft dit cohort geweigerd zonder geheime kaarten te maken.

Deze loop mag onbeperkt doorontwikkelen binnen het nachtbudget, maar haar
openbare scores zijn geen promotiebewijs. Herhaald proberen op dezelfde
toets zou vroeg of laat een vals positief opleveren. De vorige mislukte
G4-proef blijft meetellen als poging; vóór de volgende private toets moet een
nieuwe evaluator de nieuwe routekeuze kunnen valideren en vooraf een
haalbare correctie voor de opeenvolgende toetsen vastleggen. Een grotere
kindcohort kan nodig zijn zodra een strengere p-grens met vijf kinderen
onhaalbaar wordt. Zonder zo'n toets blijft G3 de officiële overlevende.

## Quotients

- IQ: task competence on held-out seeds (aggregate IQM).
- EQ: honesty, restraint and engagement from judged tasks (needs judged task data; not measurable on pure RL tasks).
- FQ: reflex sensitivity and specificity on labelled hazard scenarios, calibration of physical-rule probabilities, simulated physical tasks.
- Quotient = 100 + 15 z against a frozen, trained reference generation with real spread across seeds.
- Toddler is scored on IQ, EQ and FQ; Genie (an operator decision of 2026-10-03: a separate agent that will live in the fieldintelligence organisation, outside this repository, and uses Toddler as a training dependency; no public repository yet) on IQ only.

## Rules that apply to every step

- No synthetic or mock data as evidence; public datasets with recorded provenance only.
- Several seeds, never one run; report the variance.
- A control that gets the same mechanism without the knowledge, whenever a mechanism is claimed to help.
- Register the protocol of a follow-up before running it; no retuning until something becomes significant.
- At most half the CPU cores; GPU only through the governor; experiments that finish within three hours.
