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
| `toddler/provenance.py` | Source and consent register: no synthetic data, consent for data about people | 1, 22 |

Not in this version: the credential lifecycle module (identities, OpenBao vault, official key-creation APIs, rotation) is written locally and awaits explicit operator approval before it is pushed.

## Run the tests

```bash
python3 -m pytest -q
```

knitweb is loaded from its source tree; set `TODDLER_KNITWEB_SRC` if it is not at `/media/knight2/EDS2/projects/knitweb/src`.
Tests use real public data only (Zachary's karate club, the Wisconsin breast-cancer dataset) or explicit small matrices.

## Review workflow

From v0.1 on, changes are committed only after review, so every change carries reviewer feedback that is either applied or recorded.
