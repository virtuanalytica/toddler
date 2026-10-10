# Toddler features

Status legend: **delivered** = on `main` (v0.1.0, pushed 2026-10-03); **ready for review** = implemented and tested locally, committed only after review; **backlog** = specified, not built.
Reference document: [`docs/whitepaper/toddler-whitepaper-en.pdf`](whitepaper/toddler-whitepaper-en.pdf).
The 37-page revision adds the [five-year execution roadmap](roadmap/FIVE_YEAR_EXECUTION_2027_2031.md) and a paired work-pilot scorer. Its companion [business plan](businessplan/business-plan-2027-2031.pdf) and [technical paper](technicalpaper/technical-paper-g0-g4.pdf) are reproducible from Markdown sources. The [repo gap audit](roadmap/REPO_GAP_AUDIT_20261011.md) identifies missing independent evidence. Financial figures are planning assumptions, not observed sales.

## External benchmark and agent roadmap (9 October 2026)

The active model and specialist benchmark contract is in
[virtualv_llm](https://github.com/virtuanalytica/virtualv_llm). It keeps
software development and data in separate tables, with coder, reviewer,
architect, debugger, data engineer, data analyst, data architect and data
steward as distinct roles. A 16-item private pilot exists, with only two
items per role. It calibrates the scorers and does not rank agents or justify
a new Toddler generation.

| Proposed feature | Evidence gate | Current status |
|---|---|---|
| Route local models by task, latency and GPU-board energy | Compare the oracle ceiling, best single model and random routing on the same private items | Planned; no Toddler model router promoted |
| Add Teacher plans and reviews to ClaudeClaw | Paired, independently verified software tasks against ordinary ClaudeClaw workers | Not proven |
| Add code, data, game development and game interaction specialists | Separate specialist tests and individual promotion gates | Benchmark design in progress; agents not certified |
| Use JEV and CLM in a fast reflex | Hardware limits first, then measured model risk decisions and independent safety checks | Four JEV questions and software reflex exist; CLM adapter and physical validation open |

The [current-brain audit](design/CURRENT_BRAIN_20261009.md) separates the
surviving G3 policy router from the LLM mixture, records the JEV/CLM gaps and
distinguishes Qwen3.8-27B TP2 from Flash-Next TP2.

The full Qwen3.8 1Cat-vLLM TP2 run measured repetition loops under 1.5.0.
Version 1.5.1 passed all nine public stability checks and improved the separate
private eight-task and ten-specialist scores; its E4M3 configuration completed
B1/B4/B16 throughput probes. Claude Haiku 5.5 closed the 16-item cloud-safe
software/data pilot with 8/8 on each track. That pack is closed for later
local candidates, including 1Cat 1.5.1.
No individual model score establishes that **Toddler + Teacher + agent on
ClaudeClaw** makes better software. That claim requires a fresh independent
holdout and at least 73 paired tasks per software role. The public
[Toddler page](../site/toddler/index.html) uses the same measured/planned split.

The previous RL generation **G3-recombined** is a frozen task-expert router
distinct from the planned LLM model router above. It passed two secret-seed
gates and a [signed lineage review](learn/G3_PROMOTION_20261009.md). Its
[runtime cost probe](learn/G3_RUNTIME_BENCHMARK_20261009.md) verified all
source weights and measured CPU latency and storage; CPU energy is not
attributable on the shared host. As of 11 October, the later **G4-search**
cohort has five official children and a `promoted` private-audit status after
two independent 30-seed navigation comparisons and verified lineage. This
does not establish cognitive or software-agent competence.

## Delivered (v0.1.0)

| Feature | Module | Tests |
| ------- | ------ | ----- |
| Hard STOP rules that veto before utility | `toddler/stop.py` | test_objective |
| Reward, weekly profit, energy and compute cost, decision rule with hand-off | `toddler/objective.py` | test_objective |
| Module-graph analysis: atlas, size normalisation, sign test, clustering coefficients, similarity, consensus Louvain, null model, assignment confidence | `toddler/structure.py` | test_structure |
| Behaviour evaluation: judge-only T-scores, two axes, horizon missingness, Welch extremes, nested CV with mRMR + SVM-RFE, sensitivity/specificity, CCA + Bonferroni, outlier robustness, replication-gated promotion | `toddler/evaluation.py` | test_evaluation |
| Pulse relay: local vs knitweb peer by energy and verified cost | `toddler/relay.py` | test_relay |
| Reflex fast path: hard limits over Jev probabilities, late/missing answers stop | `toddler/fastpath.py` | test_fastpath_provenance |
| Source and consent register (no synthetic data) | `toddler/provenance.py` | test_fastpath_provenance |
| Design docs: brain architecture and the full Wee et al. 2017 mapping | `docs/design/` | - |

## Ready for review (not yet committed)

| Feature | Module | Tests |
| ------- | ------ | ----- |
| Specialisation per function-house role: Jev rules, experts, mixture-of-models routing, monotonic safety | `toddler/specialize.py` | test_specialize |
| Company guardrails with owner | `toddler/specialize.py` | test_audit |
| Hash-chained audit trail with prune anchor, secret refusal and scoped views | `toddler/audit.py` | test_audit |
| Deployment configuration `toddler-config/v1` | `toddler/config.py` | test_config |
| Configurator GUI (validation mirrored in Python) | `gui/index.html` | Playwright end-to-end run |
| Knowledge base: 36 AI/NN concepts fetched with hash and licence | `toddler/knowledge/concepts.py`, `corpus.py` | test_knowledge (hashes, structure) |
| Woven graph (concepts + 25 study steps + code); LightRAG DB when `TODDLER_GITNEXUS_BUILDER` is set | `toddler/knowledge/weave.py` | test_knowledge (all 25 rows linked, edited corpus detected) |
| knitweb synaptic bundle of the woven graph; unsigned output is `*.synaptic.unsigned`, OriginTrail publication deferred until signed | `toddler/knowledge/publish.py` | test_knowledge (round trip, signed/unsigned) |
| Whitepaper (37-page revision), business plan and technical paper | `docs/whitepaper/`, `docs/businessplan/`, `docs/technicalpaper/` | PDF render, source regeneration and forecast checks |
| Jev client (TypeSafe System One contract, strict parsing, no retries in the reflex) | `toddler/jev.py` | test_jev |
| Credential lifecycle: identities per provider, OpenBao vault, official key APIs, rotation | `toddler/credentials/` | test_credentials (held back: needs explicit operator approval to push) |

## Backlog

| Item | Blocker / next step |
| ---- | ------------------- |
| Deploy GUI to `www.knitweb.art/toddler/` | Deploy method and access for the knitweb.art server (shared with the molgang lane) |
| Deploy explainer page to `www.5mart.ml/toddler/` | TransIP SFTP access (FIVEMART_* environment) |
| Sign and publish the synaptic bundle | Operator signs with the originator key |
| OpenBao/Infisical instance for the vault | Operator sets up the instance and token |
| Run the Jev client against the live API | `TYPESAFE_API_KEY`; for physical use a local or cached Jev to meet the 20 ms reflex budget |
| Expert registry fed from MLflow evaluations | Role evaluation sets per function-house role |
| Structure analysis on recorded Toddler versions | Needs many recorded versions; run consensus + null model per release |
| C1 robot capability in simulation (MuJoCo / LeRobot, reach and point) | GPU budget within the 50 % cap; open VLA weights download |
| C2 non-contact perception (fall detection, baby monitor) | Consented public datasets; report sensitivity and specificity apart |
| C3 hardware with a care partner | Care partner, ethics committee, CE/MDR path |
| Wire the ClaudeClaw `agents/toddler` persona to this package | After review of v0.1 |
