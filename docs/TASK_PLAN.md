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
