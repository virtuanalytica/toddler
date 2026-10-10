# Toddler

Toddler is a governed learning agent that integrates virtualpc and Alexander.
It grows capability in gated phases: phase 0 is scraping only; every later capability (account actions, robot capability, physical care) passes a deliberation judge and operator approval first.

Design: [`docs/design/brain.md`](docs/design/brain.md) (architecture) and [`docs/design/wee2017-mapping.md`](docs/design/wee2017-mapping.md) (every logic step of Wee et al. 2017 linked to a Toddler function).
The ClaudeClaw agent definition (`agents/toddler/`) lives on a local, unpublished branch `feat/toddler-agent` of the operator's ClaudeClaw checkout; it is not on github.com/fillslava/ClaudeClaw. Changes to it go through ClaudeClaw's own orchestration process (`.ai/inbox` intake).

## Modules (v0.1)

| Module | What it does | Mapping rows |
| ------ | ------------ | ------------ |
| `toddler/stop.py` | Hard STOP rules that veto actions before utility is compared | 20 |
| `toddler/objective.py` | Reward per task (judge-confirmed), weekly profit, energy and compute cost, decision rule | 2, 4 |
| `toddler/structure.py` | Module-graph analysis: atlas, size normalisation, sign-test noise filter, clustering coefficients, consensus Louvain, null model, assignment confidence | 6-13 |
| `toddler/evaluation.py` | Judge-only T-scores, two behaviour axes, horizon missingness, Welch extremes, nested CV with mRMR + SVM-RFE, sensitivity/specificity, CCA + Bonferroni, outlier robustness, replication-gated promotion | 2-4, 14-18, 23-24 |
| `toddler/relay.py` | Pulse relay: local vs knitweb peer by energy and verified cost (knitweb `required_samples`, `fraud_is_profitable`) | brain.md section 4 |
| `toddler/fastpath.py` | Reflex layer for the future physical Toddler: hard limits override Jev probabilities; late or missing answers stop | 19-21 |
| `toddler/selfheal.py` | Crash watchdog: classifies failures (CUDA/host OOM, segfault, timeout, network, assertion), retries with a strictly reduced budget, quarantines a task after N consecutive failed runs (default 3, configurable per task; fail-fast, deferred-not-crashed) and records every decision in the hash-chained audit trail. `ppo.train_guarded` falls back to the CPU after a CUDA OOM with the same step budget; `benchmark_toddler`/`measure_cell` degrade broken cells to honest 'watchdog' records |
| `toddler/resources.py` | Resource governor: only free GPU memory minus a margin, skips GPUs with foreign processes, honours a configured GPU guard/preflight/lease (fail-closed), CPU threads at most half the cores minus load | brain.md section 1 (energy/compute) |
| `toddler/learn/` | Learning toddlers: Gymnasium tasks with measured random anchors and held-out seeds, PPO budgeted in environment steps (CPU runs reproducible from seed), aggregate IQM, bootstrap CI, probability of improvement | 6, 11, 12 |
| `toddler/learn/peer.py` | Toddlers teaching toddlers: KL distillation (NULL, PR #11) and pre-registered behaviour cloning (passes, PR #12), always against a no-teacher and a random-teacher control | 14, 18 |
| `toddler/learn/pbt.py` | Population-based training: members keep a TrainState, exploit copies weights + optimiser of the best, explore multiplies lr / ent_coef by 0.8 or 1.2; pre-registered test on DoorKey-5x5 is NULL (p 0.091, PR #21) | 11, 14 |
| `toddler/quotients.py` | IQ / EQ / FQ quotients (operational metrics named by analogy, not human IQ/EQ): 100 + 15 z against a frozen reference generation with a task/seed/anchor fingerprint, bootstrap CI on IQ; FQ = physical quotient; Toddler scored on IQ+EQ+FQ, Genie on IQ only | 2, 3, 18 |
| `toddler/learn/benchmark.py` | Generation benchmarks per hardware configuration: quality on CPU (hardware-independent, checked against the training record), efficiency (latency, throughput, GPU energy) only where the governor allows it now; skipped configurations reported with reason. Chain: `scripts/build_generations.py` (weights to `$TODDLER_GENERATIONS_ROOT`, outside git) then `scripts/benchmark_generations.py` | 6, 11 |
| `toddler/learn/routing.py` | Frozen task-expert successors: complete source networks per task, hash-checked generation loading and a public-state CPU cost probe; G3-recombined survived the signed lineage review | 6, 11, 14 |
| `toddler/provenance.py` | Source and consent register: no synthetic data, consent for data about people | 1, 22 |
| `toddler/specialize.py` | Roles, company guardrails, Jev questions, experts and mixture-of-models routing; safety is monotonic (add, never remove or duplicate) | 13, 20 |
| `toddler/audit.py` | Hash-chained, immutable audit trail with JSONL persistence, secret redaction, prune anchor and scoped views | 2, 25 |
| `gui/index.html` | Configurator GUI for `toddler-config/v1` (open the file in a browser; checked by `PYTHONPATH=. python3 scripts/e2e_gui.py`) | - |
| `toddler/config.py` | `toddler-config/v1` loader with path-specific errors; Python is the authority over the GUI | - |

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
| ClaudeClaw agent definition (`agents/toddler/config.yaml`, `skills_allowlist`) | local unpublished branch `feat/toddler-agent` of the operator's ClaudeClaw checkout (not publicly resolvable) |
| Deliberation gate and judge authority (`docs/deliberation-gate.md`, `config/model-routing.yaml`) | [fillslava/ClaudeClaw](https://github.com/fillslava/ClaudeClaw) |
| Review discipline (`.claude/rules/review-discipline.md`) | [fillslava/ClaudeClaw](https://github.com/fillslava/ClaudeClaw) |
| LightRAG source policy (`src/lightrag-source-policy.ts`) | [fillslava/ClaudeClaw](https://github.com/fillslava/ClaudeClaw) |
| knitweb (`pouw`, `synaptic`, `p2p`) | the knitweb repository (see `toddler/_knitweb.py`) |

## Chat with a toddler (OMP harness + mixture of models)

A chat model can put a real toddler to work through the MCP server `toddler/learn/mcp_server.py`
(tools: `list_toddlers`, `list_tasks`, `play_episode`, `iq_probe`, `trace`). Every result is computed
by toddler code; the models only explain it.

1. Start the live mixture of models (virtualv_llm): `infra/model_serve_configs/mom-live.sh start`
   (OpenAI-compatible endpoint `http://127.0.0.1:8030/v1`, model `mom-live`).
2. OMP profile `toddler` (`~/.omp/profiles/toddler/agent/`): `models.yml` adds provider `virtualv`
   with model `mom-live`; `mcp.json` starts `python3 -m toddler.learn.mcp_server` with
   `TODDLER_GENERATIONS_ROOT=/media/knight2/EDS2/toddler-generations`.
3. Chat: `omp --profile toddler --model virtualv/mom-live` (shortcut `omp-toddler`), e.g.
   "Laat G1/t1001 doorkey5 spelen met seed 424242" or "Meet het IQ van G1/t1003".

Without a chat: `python3 -m toddler.learn.play episode G1/t1001 doorkey5 --seed 424242` and
`python3 -m toddler.learn.play iq G1/t1001`.

The [G3-recombined runtime review](docs/learn/G3_RUNTIME_BENCHMARK_20261009.md)
records the two secret-seed confirmations separately from an operational CPU
probe. The probe verifies frozen source weights and measures latency and
storage, without opening secret seeds or judging answer quality. The
[signed survivor decision](docs/learn/G3_PROMOTION_20261009.md) records the
five imported children, verified lineage, remaining limitations and
[development report](docs/learn/reports/G3-recombined_ontwikkelverslag.pdf).

## Software-agent benchmark integration

The current model comparisons and the separate software/data specialist pilot
live in [virtualv_llm](https://github.com/virtuanalytica/virtualv_llm).
The pilot measures eight roles but cannot prove that Toddler + Teacher + an
agent on ClaudeClaw improves software work. That claim needs paired tasks,
an independently controlled private holdout and at least 73 tasks per
software role. See [feature status](docs/FEATURES.md#external-benchmark-and-agent-roadmap-9-october-2026)
and the [public Toddler page](site/toddler/index.html), including a
[neuroscience-to-AI guide for the public](site/toddler/neurowetenschap.html).
The [technical brain audit](docs/design/CURRENT_BRAIN_20261009.md) and
[architecture page](site/toddler/architectuur.html) distinguish the G3 policy
router, JEV reflex, CLM shadow adapter and measured GPU mixture. The
[G4 curriculum and CLM protocol](docs/G4_CLM_SHADOW.md) documents the
earlier cognitive/navigation candidate, nightly Teacher run and local CLM
comparison. Oracle imitation solved `unlockpickup` on 45/50 public development
maps for one child. The first [five-child private ancestor trial](docs/learn/G4_ORACLE_RESULT_20261009.md)
improved the aggregate score without passing both pre-registered tests. A
later, separately controlled `G4-search` cohort passed two independent
private 30-seed navigation trials; the official private status is `promoted`
with five children and verified lineage as of 11 October 2026. This does
not promote the cognitive curriculum or establish agent or physical-work
capability.
The [plasticity research design](docs/learn/TODDLER_PLASTICITY_RESEARCH_20261010.md)
defines the next replay, modular expert and adaptation experiments. Its
dated status paragraph predates the G4-search confirmation.
The [capability atlas](knowledge/README.md) provides 10,000 stable learning
addresses and a first ten-lesson adaptive-learning path. Unreviewed slots do
not imply that Toddler has mastered those capabilities.
The [37-page white paper](docs/whitepaper/toddler-whitepaper-en.pdf)
adds the path from Toddler training to useful work, PLS/PAR evidence and an
owner-led five-year agent-economy scenario. The companion
[business plan](docs/businessplan/business-plan-2027-2031.pdf) includes the
Virtuanalytica logo, market/competitor scan, diagrams, operating and marketing
plans, and a monthly 2027 liquidity scenario. The
[technical paper](docs/technicalpaper/technical-paper-g0-g4.pdf) describes
G0–G4 lineage, data and training protocols, knowledge graph, source layout and
developer reproduction steps. The [repo gap audit](docs/roadmap/REPO_GAP_AUDIT_20261011.md)
lists the missing evidence and engineering work. The
[execution roadmap](docs/roadmap/FIVE_YEAR_EXECUTION_2027_2031.md) starts with
a sealed software-debugger pilot. Its [work-pilot scorer](scripts/score_work_pilot.py)
compares the incumbent, Toddler + Teacher, and Toddler + Teacher + specialist
on paired tasks, including accepted-work quality, review effort, energy and
direct cost. The scenario is generated by
[`scripts/forecast_toddler_business.py`](scripts/forecast_toddler_business.py);
none of its customer or revenue targets is an observed result.
The Qwen3.8 1Cat
TP2 comparison records the 1.5.0 repetition failures, the 1.5.1 quality
improvement, throughput and GPU-board energy. Haiku 5.5 closed the separate
cloud-safe model pilot at 8/8 software and 8/8 data; 1.5.1 was measured
after that pack closed and is absent from its role tables.

## Known gaps

- The capability `ATLAS` in `toddler/structure.py` lists regions that have no code yet (`perception.*`, `actuation`). `toddler/codegraph.py` (mapping step B) emits module-graph edges from the real repository: static imports between atlas regions, sizes in lines, empty regions kept at zero; `scripts/module_graph.py` measures committed revisions into `docs/design/module_graph.json`. Execution-path tracing with stop criteria (mapping row 7) is still planned.
- See issue #1 for the open review items.

## Distributed development

Anyone can propose a stronger Toddler through a fork and pull request. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the required reproducible recipe, model
and interaction disclosure, contributor trials, and the independent hidden
ancestor audit. The G3 promotion was the final originator-signed pivot by
Deve Luse; later promotions use preregistered audit trials and lineage
verification. A merged trial PR does not itself create a survivor. Contributors
with repository merge access may merge their own passing PRs; everyone else
uses maintainer review.

## Licence

Original source code, documentation and generated reports in this repository
are Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Wikipedia extracts
in `data/corpus/` remain CC BY-SA 4.0 with attribution in each file. External
trained weights, private datasets and third-party models are not distributed
or relicensed by this repository.
