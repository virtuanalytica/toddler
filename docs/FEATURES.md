# Toddler features

Status legend: **delivered** = on `main` (v0.1.0, pushed 2026-10-03); **ready for review** = implemented and tested locally, committed only after review; **backlog** = specified, not built.
Reference document: [`docs/whitepaper/toddler-whitepaper-en.pdf`](whitepaper/toddler-whitepaper-en.pdf).

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
| Knowledge base: 36 AI/NN concepts fetched with hash and licence | `toddler/knowledge/concepts.py`, `corpus.py` | coverage check in weave |
| Woven graph (concepts + 25 study steps + code), LightRAG facts DB | `toddler/knowledge/weave.py` | 0 coverage problems |
| knitweb synaptic bundle of the woven graph (unsigned) | `toddler/knowledge/publish.py` | decode round-trip |
| Whitepaper (12 pages) | `docs/whitepaper/` | build verification |
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
