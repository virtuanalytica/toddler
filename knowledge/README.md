# Toddler capability-atlas

De atlas bevat **10.000 gespecificeerde vakbekwaamheden met stabiele ID's**: 10 domeinen × 10 onderwerpen × 10 werkwijzen × 10 situaties. `CAPABILITY_INDEX.txt` en `atlas.json` leggen de volgorde en hash vast. Elke map heeft een `capability.json` met een vakinhoudelijke taak, praktijkproduct, meetmaat, kenmerkende fout, benodigde bewijsstukken en vier toetscriteria. Dit zijn **gegenereerde, nog niet beoordeelde specificaties**. `status: scaffold` en `mastery_claim: false` betekenen dat volledig onderwijs, bronnen en bewezen beheersing meestal ontbreken. De telling is geen telling van bewezen Toddler-vermogens.

Begin bij [de leerlijn adaptief leren](paths/adaptive_learning.md), de [kwaliteitseisen](QUALITY_STANDARD.md), het [onderzoek naar plasticiteit](../docs/learn/TODDLER_PLASTICITY_RESEARCH_20261010.md) en het [opslag- en kennisgraafplan](STORAGE_AND_GRAPH_PLAN_20261010.md). Die route verbindt primaire literatuur aan hands-on werk in de bestaande Toddler-code. Een PhD-niveau vraagt originele onderzoeksvragen, reproduceerbare resultaten en kritische beoordeling; een expert practitioner moet daarnaast de systemen onder echte grenzen kunnen bouwen, meten, herstellen en uitleggen. De atlas organiseert dat werk, maar verleent geen graad of kwaliteitsclaim.

Zie [10.000 vakbekwaamheden](COMPETENCY_CATALOG.md) voor de inhoudelijke opbouw, beoordelingsgrens en Teacher-overdracht.

De [zes uitgewerkte voorbeelden](SIX_KNOWLEDGE_EXAMPLES.md) en de [kennisopbouwpipeline](../docs/KNOWLEDGE_COLLECTION_PIPELINE.md) tonen de eerste bronbundels en een cronveilige dagwachtrij. `knowledge/topic_sources.jsonl` scheidt gecontroleerde primaire links van de onbeoordeelde zoekresultaten in de runtime.

Gebruik:

```bash
python3 scripts/capability_atlas.py validate
python3 scripts/capability_atlas.py generate
python3 scripts/capability_lookup.py continual_learning/experience_replay/implement__simulated_task
python3 scripts/capability_atlas.py export --out /tmp/toddler-competencies.jsonl
```

`generate` maakt ontbrekende slots en werkt hun deterministische vakbekwaamheidsspecificatie bij zonder bestaande lesstatus of bewijzen te verwijderen. `validate` controleert identiteiten, aantallen, unieke specificaties en de canonieke index. `export` geeft Teacher alle 10.000 publieke specificaties in JSONL; de private toetsvragen en antwoorden staan daar niet in. Uitgewerkte lessen komen als `lesson.md` in hun capability-map; reviewers mogen een slot pas voorbij `scaffold` brengen als bron, oefening en toets volgens de kwaliteitseisen zijn beoordeeld. Een model of Toddler-generatie erft nooit automatisch een capability-status uit dit catalogusbestand.

De vier assen zijn een **ordeningsstelsel**, geen wetenschappelijke ontologie. Sommige combinaties kunnen na vakinhoudelijke review onzinnig blijken. Markeer die als `retired` met reden; hergebruik nooit hun ID. De 10.000 specificaties zijn een toetsbare curriculumbacklog voor crowdsourcing, geen 10.000 gevalideerde lessen.
