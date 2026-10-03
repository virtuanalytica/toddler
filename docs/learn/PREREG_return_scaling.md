# Pre-registration: return scaling in PPO

Registered 2026-10-03, before the confirmatory run. Exploratory evidence that motivated it
(seeds 1-2, 150k steps, not part of the test): Acrobot normalised -0.002 / -0.002 without
scaling vs 1.034 / 1.033 with it; CartPole 0.322 / 0.995 vs 1.055 / 1.055; MountainCar 0.0 in
both conditions.

- Hypothesis: `PPOConfig(scale_rewards=True)` gives a higher normalised held-out score than
  `scale_rewards=False` at an equal budget of 150 000 environment steps.
- Tasks: Acrobot-v1 and CartPole-v1, tested separately. MountainCar-v0 is excluded: both
  conditions score 0 (sparse reward; needs a different exploration method, not this change).
- Seeds: 21, 22, 23, 24, 25 (not used in the exploration). Evaluation on the 30 held-out seeds,
  greedy actions, CPU.
- Test: one-sided Mann-Whitney U per task, alpha 0.05, plus probability of improvement.
- Decision: the default becomes `scale_rewards=True` only if BOTH tasks pass. Existing
  generations keep their recorded method version (`scale_rewards=False`) and stay reproducible.
- Everything is reported, including a failure.
