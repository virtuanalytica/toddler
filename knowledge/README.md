# Toddler capability-atlas

De atlas bevat **10.000 mappen met stabiele capability-ID's**: 10 domeinen × 10 onderwerpen × 10 werkwijzen × 10 situaties. `CAPABILITY_INDEX.txt` en `atlas.json` leggen de volgorde en hash vast. Elke map heeft een `capability.json`. De naam benoemt een leerdoel; `status: scaffold` en `mastery_claim: false` betekenen dat er nog geen uitgewerkt onderwijs of bewijs is. De telling van mappen is **geen** telling van bewezen vermogens.

Begin bij [de leerlijn adaptief leren](paths/adaptive_learning.md), de [kwaliteitseisen](QUALITY_STANDARD.md) en het [onderzoek naar plasticiteit](../docs/learn/TODDLER_PLASTICITY_RESEARCH_20261010.md). Die route verbindt primaire literatuur aan hands-on werk in de bestaande Toddler-code. Een PhD-niveau vraagt originele onderzoeksvragen, reproduceerbare resultaten en kritische beoordeling; een expert practitioner moet daarnaast de systemen onder echte grenzen kunnen bouwen, meten, herstellen en uitleggen. De atlas organiseert dat werk, maar verleent geen graad of kwaliteitsclaim.

Gebruik:

```bash
python3 scripts/capability_atlas.py validate
python3 scripts/capability_atlas.py generate
```

`generate` maakt ontbrekende slots en bewaart bestaande metadata. `validate` controleert identiteiten, aantallen en de canonieke index. Uitgewerkte lessen komen als `lesson.md` in hun capability-map; reviewers mogen een slot pas voorbij `scaffold` brengen als bron, oefening en toets volgens de kwaliteitseisen zijn beoordeeld. Een model of Toddler-generatie erft nooit automatisch een capability-status uit dit catalogusbestand.

De vier assen zijn een **ordeningsstelsel**, geen wetenschappelijke ontologie. Sommige combinaties hebben mogelijk geen zinnig lesdoel. Markeer die bij inhoudelijke review als `retired` met reden; hergebruik nooit hun ID. Het doel van de 10.000 mappen is stabiele adressering voor crowdsourcing, niet het fabriceren van 10.000 lessen.
