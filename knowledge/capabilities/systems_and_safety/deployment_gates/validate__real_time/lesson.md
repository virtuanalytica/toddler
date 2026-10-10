# Adaptieve policy veilig inzetten

**Bronnen:** [Toddler G3 runtime-metingen](../../../../../docs/learn/G3_RUNTIME_BENCHMARK_20261009.md), [G4 CLM-schaduwproef](../../../../../docs/G4_CLM_SHADOW.md) en [Laroche e.a., Safe Policy Improvement with Baseline Bootstrapping](https://proceedings.mlr.press/v97/laroche19a.html). De papergarantie heeft aannames over batch-RL en geldt niet automatisch voor Toddler of fysieke actuatie.

**Theorieopdracht:** onderscheid modelinferentie, endpoint-latency, fysieke actuatordeadline en worst-case foutpad. Leg vast welke state bij een update atomair moet wisselen en hoe rollback lineage-verwijzingen bewaart.

**Praktijk:** voer op een niet-actieve kandidaat een latency- en foutinjectieproef uit: timeout, corrupte gewichten, ontbrekende expert, hoge onzekerheid en terugval op de bewezen ouder. Meet p50/p95/p99, geheugen en energie per beslissing. **Toets:** een reviewer kan de checkpointhash verifiëren en bij elke fout de oude policy herstellen zonder private toetsdata te laden.
