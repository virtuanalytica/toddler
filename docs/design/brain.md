# Toddler brain architecture

Operator definition (2026-10-03): Toddler's brain holds part of the logic, the knitweb P2P capabilities with Pulse behaviours, and compute-cost / energy-based trade-offs, next to the loss and reward functions.
For the physical part (later) it uses vision theory with cameras and lidar, KUKA-grade precision, Unitree-style AI to understand the physical world, and Jev to evaluate physical rules very fast for the heuristic layer (fast decisions and reaction speed).

Nothing in this document grants a capability.
Every component beyond phase 0 (scraping) passes the deliberation gate and operator approval before a code path exists.

## 1. The objective, in one place

Toddler chooses an action `a` (including "do nothing" and "ask a human") by maximising

```
U(a) = E[reward(a)] + w_profit * E[profit(a)]
       - w_energy * energy_J(a) - w_compute * compute_cost_eur(a)
       - w_risk * P(failure | a) * severity(a)
subject to: STOP(a) == false   (hard constraints, never traded against reward)
```

- **Loss function**: what a learned component minimises during training (for example cross-entropy for a classifier, a policy-gradient loss for a robot policy). It shapes the weights; it is not the decision rule.
- **Reward function**: the per-task learning signal (+1 done and confirmed by the judge, +0.2 honest "I don't know", +0.1 new useful knowledge, -1 wrong, -5 claiming done when not done, minus cost). The judge, not Toddler, confirms "done".
- **Profit function**: the weekly system-level account (value delivered minus energy, compute, human review time and risk). It decides whether a phase continues; it does not train Toddler step by step.
- **Energy / compute cost**: every action carries an estimated cost in joules and euros. Reference point used in the explainer page: a human brain runs on about 20 W, AlphaGo used about 1 MW.
- **STOP rules**: never show a secret, never act on a person without a present caregiver, never buy, send or log in without approval. These are constraints, not penalties, so no amount of reward can buy them.

All weights (`w_*`) start as documented placeholders and are only changed after a measured, replicated improvement (Wee et al. 2017 mapping rows 11, 12, 24).

## 2. Logic layer

Deterministic rules that do not need learning: the STOP rules, the phase allowlist (`config.yaml` `skills_allowlist`), provenance and consent checks, and the hand-off rule (low confidence -> ask a human).
This layer runs before any learned component and can veto it (mapping row 20: the inhibition layer is wired to every action).

## 3. knitweb P2P capabilities

Reused from `/media/knight2/EDS2/projects/knitweb/src/knitweb` (no new P2P code):

| Need                                 | knitweb module                                                                                                                                                                                              |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Talk to peers, replicate feeds       | `p2p/node.py` (`AsyncioP2PNode`, `StaticPeerBook`, `FeedReplica`), `p2p/wire.py`                                                                                                                            |
| Package knowledge for sharing        | `synaptic/bytecode.py` (`compile_bundle`, `sign_bundle`, `verify_bundle`)                                                                                                                                   |
| Prove where knowledge came from      | `fabric/provenance.py`, `fabric/attest.py`                                                                                                                                                                  |
| Offload work to a peer and verify it | `pouw/job.py` (`execute`, `verify`), `pouw/committee.py` (`select_committee`), `pouw/sampling.py` (`catch_probability`, `required_samples`), `pouw/escrow.py`, `pouw/collateral.py` (`fraud_is_profitable`) |
| Publish the woven theory+code graph  | `synaptic/origintrail.py` (publication layer, operator decision)                                                                                                                                            |

## 4. Pulse behaviours

From the operator's Pulse definition: a trustless relay to compute and technique, node-to-node sharing of resources, and utility-driven value.
In Toddler that becomes four behaviours, each feeding the energy/compute term of `U(a)`:

1. **Relay to compute**: if a peer can do a job cheaper or greener than the local box, offer it as a PoUW job and pay only after verification (`pouw/verify`).
2. **Share node-to-node**: knowledge bundles Toddler produced are shared as signed synaptic bundles, never raw scraped pages.
3. **Network weather**: read peer load and price before choosing local versus remote compute.
4. **Pay per served byte**: account the real cost of what was fetched or served, so the profit function sees it.

Verification is never skipped: a cheap but unverified result scores as a failure (`fraud_is_profitable` guards the collateral side).

## 5. Physical part (future, phases C1-C3)

Two speeds, mirroring the paper's early-safety / later-control ordering (mapping rows 19-21):

- **Fast path (reflex / heuristic, milliseconds)**: Jev answers many typed physical-rule questions in parallel ("will this grasp slip?", "is a person inside the safety zone?", "is this force above the limit?") and returns probabilities with uncertainty. Combined with hard safety limits, this decides stop, slow down or continue. Jev is used as a fast feature generator, never as the only authority, the same way the Jev finance experiment uses it (`virtualpc-jev-finance/docs/JEV-FINANCE-EXPERIMENT.md`).
- **Slow path (deliberate, seconds)**: planning with the world model and the judge, only when the fast path says it is safe to think.

Components:

| Capability                       | Approach                                                                                                                                                       | Safety reference                                   |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| Seeing                           | Cameras (RGB-D) + lidar, sensor fusion into one 3D scene                                                                                                       | Report misses and false alarms separately (row 16) |
| Precise movement                 | KUKA-grade motion control: calibrated kinematics, force/torque limits, repeatable trajectories                                                                 | ISO 10218 and ISO/TS 15066 (collaborative robots)  |
| Understanding the physical world | Unitree-style learned policies (reinforcement learning in simulation, sim-to-real), open VLA models (OpenVLA, pi0, GR00T) fine-tuned, not trained from scratch | Simulation first (MuJoCo / Isaac / LeRobot)        |
| Fast physical rules              | Jev parallel typed questions -> probabilities for the reflex layer                                                                                             | Hard limits always override probabilities          |

Contact with babies is out of scope; elderly care starts with non-contact help and always with a caregiver present.

## 6. How the parts connect

```
inputs (web now, sensors later)
   -> logic layer (STOP, allowlist, provenance)          [veto]
   -> fast path: Jev physical-rule probabilities          [reflex]
   -> slow path: planner + world model + memory graph     [deliberate]
   -> choose a maximising U(a) under STOP                 [decision]
   -> act locally, or relay to a knitweb peer (Pulse)     [execution]
   -> judge confirms -> reward; weekly -> profit          [learning signal]
   -> everything appended to the hash-chained audit_log   [oversight]
```
