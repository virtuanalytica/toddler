# Van 10.000 adressen naar 10.000 vakbekwaamheden

**Status 11 oktober 2026:** 10.000 unieke, machineleesbare *vakbekwaamheidsspecificaties*; zestien brongebonden lesconcepten, waarvan zes nieuwe kennisbundels; nul claims dat een Toddler alle vakbekwaamheden beheerst. Het canonieke ID-register en zijn SHA-256 zijn ongewijzigd. De combinatie is 100 vakonderwerpen × tien professionele handelingen × tien werksituaties. Een vakbekwaamheid is hier een toetsbare uitkomst, geen diploma of attest.

## Wat elk item bevat

`knowledge/capabilities/<domein>/<onderwerp>/<handeling>__<situatie>/capability.json` bevat nu `competency`:

1. **Vakinhoudelijke kern:** een concreet product, een observeerbare maat en een typische denk- of uitvoeringsfout. Deze 100 kernen zijn afzonderlijk geformuleerd in `scripts/competency_specs.py`.
2. **Beroepshandeling:** definiëren, verklaren, afleiden, bouwen, reproduceren, diagnosticeren, vergelijken, ontwerpen, valideren of onderwijzen. Elke handeling vraagt een ander product en een eigen functionele check.
3. **Werksituatie:** uitgewerkt voorbeeld, simulatie, velddata, weinig voorbeelden, taakvolgorde, verdelingsverschuiving, rekenbudget, menselijke review, onafhankelijke holdout of realtime gebruik. Iedere situatie voegt een meetbare randvoorwaarde toe.
4. **Bewijs en rubric:** vier noodzakelijke criteria voor begrip, handeling, meting en kritiek; plus vereiste artefacten en een onafhankelijke nieuwe toetscasus. Er staan geen verborgen vragen of antwoorden in de publieke atlas.

Voorbeeld: `reasoning_and_language/numeracy/implement__few_shot` vraagt een uitvoerbare eenheidsbewuste rekenhulp, een vooraf vastgelegd klein voorbeeldenbudget, exacte antwoorden én correcte eenheden, plus een negatieve controle voor plausibele maar foutieve orde van grootte. `systems_and_safety/deployment_gates/validate__real_time` vraagt een onafhankelijke validatie van een versiegebonden promotie- en rollbackgate met p95-latentie, timeout- en terugrolspoor.

## Kwaliteitsgrens

De generator levert een **curriculumcontract**: wat moet worden gemaakt en getoond. Hij levert geen primaire literatuur, uitvoerbare implementatie, videomateriaal, gevalideerde les, geheime toets, beoordelingsuitslag of PhD-niveau. Dat is apart menselijk en experimenteel werk. `status: scaffold` blijft staan totdat een brongedekte `lesson.md` en praktijkopdracht bestaan. De zestien `lesson_draft`-records blijven lesconcepten totdat een onafhankelijke vakreview ze heeft beoordeeld. `competency.status: generated_unreviewed` is expliciet; er is geen automatische promotie.

Een inhoudelijke reviewer moet een gegenereerde specificatie afkeuren of aanscherpen als de handeling/situatie voor het onderwerp niet zinvol is, de maat geen kwaliteit meet, de negatieve controle niet discrimineert of veiligheidsgrenzen ontbreken. Afgekeurde ID's krijgen `retired` met reden. Per geaccepteerde les vereist [QUALITY_STANDARD.md](QUALITY_STANDARD.md) een primaire bron, oefening, reproduceerbaarheid en onafhankelijke beoordeling. Een Toddler krijgt pas een individueel auditrecord na een verzegelde, nieuwe toets; geen catalogusitem draagt `mastery_claim: true`.

## Gebruik door Teacher en bijdragers

```bash
python3 scripts/capability_atlas.py validate
python3 scripts/capability_lookup.py reasoning_and_language/numeracy/implement__few_shot
python3 scripts/capability_atlas.py export --out /tmp/toddler-competencies.jsonl
```

De export bevat alle 10.000 `capability_id`, `lesson_status` en `competency`-records in canonieke volgorde. Teacher kan er leerplannen en reviewtaken uit kiezen. Alleen brongebonden en gecontroleerde lessen mogen als oefenmateriaal worden aangeboden; de 10.000 specificaties zijn geen answer-keyed trainingsset. Houd de promotietoetsen apart, inclusief prompts, seeds en antwoorden. Publiceer bij een bijdrage bron, licentie, code- en datahash, praktijkresultaat en reviewstatus; verander nooit een bestaand ID.

`validate` controleert alle 10.000 bestanden, de 100 vakinhoudelijke kernen, de vier assen, de indexhash en dat elke specificatie uniek en ongewijzigd is. Dit is structurele controle; vakinhoudelijke juistheid vergt nog review. De bronstatus is daarom zichtbaar in elk record en in de atlas-samenvatting.
