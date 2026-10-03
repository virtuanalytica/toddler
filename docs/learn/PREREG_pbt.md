# Pre-registration: population-based training on DoorKey-5x5

Registered 2026-10-03, before the confirmatory run. Code: `toddler/learn/pbt.py` (members keep a
`ppo.TrainState`, so intervals continue one run; PR #20).

Exploratory evidence that chose the budget (populations seeds 1-2, control arm only, not part of
the test): 80k steps per member gives -0.03 / 0.261, 100k gives 0.964 / 0.706. 80k is the regime
between floor and ceiling.

- Hypothesis: a PBT population returns a better toddler than a control population with the same
  schedule and the same number of environment steps.
- Task: `doorkey5` (MiniGrid-DoorKey-5x5-v0, PR #19), PPO defaults (return scaling on).
- Population: 4 members, 8 intervals of 10 000 steps each (80 000 steps per member, 320 000 per
  population, equal in both arms).
- PBT arm: after every interval except the last, the member with the lowest training score
  (mean return of its last 20 training episodes) copies the weights and optimiser state of the
  highest, then multiplies learning rate and entropy coefficient by 0.8 or 1.2 (seeded).
  Control arm: identical, without exploit/explore.
- Output per population: the member with the best training score after the last interval,
  normalised on the 30 held-out evaluation seeds (never seen by the selection).
- Population seeds: 21, 22, 23, 24, 25 in each arm.
- Test: one-sided Mann-Whitney U (PBT > control), alpha 0.05; also probability of improvement.
- Decision: PBT is reported as helping only if p < 0.05. Everything is reported, including a NULL.
