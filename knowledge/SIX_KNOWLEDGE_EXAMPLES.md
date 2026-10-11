# Zes uitgewerkte kennisbundels

Deze zes voorbeelden tonen hoe de 10.000 vakbekwaamheden vanuit onderwerpbronnen naar een concrete les en praktijktoets groeien. De bronnen zijn tegen de primaire publicaties of officiële documentatie gecontroleerd op 11 oktober 2026. Alle zes zijn `lesson_draft`; ze zijn nog niet onafhankelijk vakinhoudelijk beoordeeld en geen Toddler beheersingsbewijs.

| Vakbekwaamheid | Kennisbron | Praktijkproef en faalcontrole |
|---|---|---|
| [Beloningsontwerp diagnosticeren](capabilities/reinforcement_learning/reward_design/diagnose__simulated_task/lesson.md) | ICML-onderzoek over constrained RL en reward hacking | Proxyreturn en echte taakscore uiteen laten lopen in een gezaaide gridworld; overtredingen afzonderlijk tellen. |
| [Geheugenherkomst ontwerpen](capabilities/memory_systems/memory_provenance/design__field_data/lesson.md) | W3C PROV-O en PROV-DM | Feiten aan bronhash, extractierun en verantwoordelijke koppelen; correctie en verwijdering door de afleidingsgraaf volgen. |
| [Rekenen met eenheden bouwen](capabilities/reasoning_and_language/numeracy/implement__few_shot/lesson.md) | BIPM SI Brochure en NIST | Calculator met maximaal vijf oefenvoorbeelden; onverenigbare eenheden en schaalfouten afwijzen. |
| [Meertalig leren vergelijken](capabilities/reasoning_and_language/multilingual_learning/compare__distribution_shift/lesson.md) | FLORES-200 en NLLB | Twee modellen gepaard vergelijken op nieuwe domeinzinnen; betekenis, getallen en register apart beoordelen. |
| [GUI-gronding valideren](capabilities/multimodal_agents/gui_grounding/validate__independent_holdout/lesson.md) | OSWorld-paper en officiële code | Een eigen afgeschermde, wisselende GUI-taak; stale coördinaten en onomkeerbare acties detecteren. |
| [Energie per antwoord reproduceren](capabilities/evaluation_and_causality/energy_accounting/reproduce__compute_budget/lesson.md) | NVIDIA DCGM en Linux powercap | GPU-board Wh apart van CPU-zone en systeemenergie rapporteren; resets en eenheidsfouten uitfilteren. |

De gedeelde bronregistratie staat in [topic_sources.jsonl](topic_sources.jsonl). [De productielijn](../docs/KNOWLEDGE_COLLECTION_PIPELINE.md) behandelt de resterende onderwerpen en vakbekwaamheden met een dagelijkse, auditeerbare wachtrij. De gehashte bronlinks zijn geen garantie dat ieder afgeleid lesdoel correct is; dat vraagt onafhankelijke review per les.
