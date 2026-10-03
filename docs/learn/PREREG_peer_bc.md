# Pre-registration: behaviour-cloning warm start (follow-up to PR #11)

Registered before running, 2026-10-03.

- **Question**: does a trained teacher help a student when its knowledge is passed by behaviour cloning (supervised learning on the teacher's actions) before self-reinforcement learning?
- **Groups** (5 student seeds each: 11-15): real teacher, no teacher, random (untrained) teacher.
- **Budget** (equal for all): 30,000 environment steps per student. With a teacher: 10,000 steps of teacher rollouts for cloning (counted against the budget), then 20,000 PPO steps. Without a teacher: 30,000 PPO steps.
- **Teacher**: same as PR #11 (CartPole, 150,000 steps, seed 2).
- **Cloning**: cross-entropy on teacher greedy actions, 20 epochs, minibatch 256, lr 1e-3.
- **Metric**: normalised score on the 30 held-out seeds, CPU, greedy.
- **Decision rule** (unchanged): teaching helps only if the real teacher beats both controls (one-sided Mann-Whitney U, alpha 0.05).
- **No retuning** after seeing the result; any further variant is a new pre-registration.
