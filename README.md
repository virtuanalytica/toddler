# Toddler

Toddler is a governed learning agent that integrates virtualpc and Alexander.
It grows capability in gated phases: phase 0 is scraping only; every later capability (account actions, robot capability, physical care) passes a deliberation judge and operator approval first.

Design: [`docs/design/brain.md`](docs/design/brain.md) (architecture) and [`docs/design/wee2017-mapping.md`](docs/design/wee2017-mapping.md) (every logic step of Wee et al. 2017 linked to a Toddler function).
The ClaudeClaw agent definition lives on branch `feat/toddler-agent` of the fillslava ClaudeClaw fork (`agents/toddler/`).

## Modules (v0.1)

| Module | What it does | Mapping rows |
| ------ | ------------ | ------------ |
| `toddler/stop.py` | Hard STOP rules that veto actions before utility is compared | 20 |
| `toddler/objective.py` | Reward per task (judge-confirmed), weekly profit, energy and compute cost, decision rule | 2, 4 |
| `toddler/structure.py` | Module-graph analysis: atlas, size normalisation, sign-test noise filter, clustering coefficients, consensus Louvain, null model, assignment confidence | 6-13 |
| `toddler/evaluation.py` | Judge-only T-scores, two behaviour axes, horizon missingness, Welch extremes, nested CV with mRMR + SVM-RFE, sensitivity/specificity, CCA + Bonferroni, outlier robustness, replication-gated promotion | 2-4, 14-18, 23-24 |
| `toddler/relay.py` | Pulse relay: local vs knitweb peer by energy and verified cost (knitweb `required_samples`, `fraud_is_profitable`) | brain.md section 4 |
| `toddler/fastpath.py` | Reflex layer for the future physical Toddler: hard limits override Jev probabilities; late or missing answers stop | 19-21 |
| `toddler/resources.py` | Resource governor: only free GPU memory minus a margin, skips GPUs with foreign processes, honours a configured GPU guard/preflight/lease (fail-closed), CPU threads at most half the cores minus load | brain.md section 1 (energy/compute) |
| `toddler/learn/` | Learning toddlers: Gymnasium tasks with measured random anchors and held-out seeds, PPO budgeted in environment steps (CPU runs reproducible from seed), aggregate IQM, bootstrap CI, probability of improvement | 6, 11, 12 |
| `toddler/quotients.py` | IQ / EQ / FQ quotients (operational metrics named by analogy, not human IQ/EQ): 100 + 15 z against a frozen reference generation with a task/seed/anchor fingerprint, bootstrap CI on IQ; FQ = physical quotient; Toddler scored on IQ+EQ+FQ, Genie on IQ only | 2, 3, 18 |
| `toddler/provenance.py` | Source and consent register: no synthetic data, consent for data about people | 1, 22 |

Not in this version: the credential lifecycle module (identities, OpenBao vault, official key-creation APIs, rotation) is written locally and awaits explicit operator approval before it is pushed.

## Install and run the tests

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest -q
```

knitweb (needed by `toddler/relay.py`) is located by `toddler/_knitweb.py`: an installed `knitweb` package, the `TODDLER_KNITWEB_SRC` environment variable, or a sibling checkout at `../knitweb`.
Without knitweb the relay tests are skipped, so CI runs without it.
Tests use real public data only (Zachary's karate club, the Wisconsin breast-cancer dataset) or explicit small matrices.

## Review workflow

From v0.1 on, changes are committed only after review, so every change carries reviewer feedback that is either applied or recorded.

## External references

Several documents mention artefacts that live outside this repository:

| Artefact | Where |
| -------- | ----- |
| ClaudeClaw agent definition (`agents/toddler/config.yaml`, `skills_allowlist`) | branch `feat/toddler-agent` of the fillslava ClaudeClaw fork ([fillslava/ClaudeClaw](https://github.com/fillslava/ClaudeClaw)) |
| Deliberation gate and judge authority (`docs/deliberation-gate.md`, `config/model-routing.yaml`) | [fillslava/ClaudeClaw](https://github.com/fillslava/ClaudeClaw) |
| Review discipline (`.claude/rules/review-discipline.md`) | [fillslava/ClaudeClaw](https://github.com/fillslava/ClaudeClaw) |
| LightRAG source policy (`src/lightrag-source-policy.ts`) | [fillslava/ClaudeClaw](https://github.com/fillslava/ClaudeClaw) |
| knitweb (`pouw`, `synaptic`, `p2p`) | the knitweb repository (see `toddler/_knitweb.py`) |

## Known gaps

- The capability `ATLAS` in `toddler/structure.py` lists modules that have no code yet (`perception.*`, `actuation`, `memory.graph`, `oversight.*`). Nothing emits module-graph edges from the real repository yet; that producer is planned (mapping step B), so the structure functions are validated on public data only.
- See issue #1 for the open review items.
