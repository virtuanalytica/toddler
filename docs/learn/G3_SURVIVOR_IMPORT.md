# G3-recombined: overlevingspoort en import

**Status op 9 oktober 2026:** G3-recombined voldoet op twee onafhankelijke,
vooraf vastgelegde geheime seedsets aan de statistische promotiepoort. G2
blijft de officiële generatie totdat een menselijke reviewer het bevroren
[reviewpakket](G3_RECOMBINED_REVIEW.json) tekent en de kandidaat is geïmporteerd.
De eerdere G3-sp, G3-trunk en G3-scratch blijven uitgestorven zijtakken. Hun
gewichten zijn alleen als bevroren bronexperts in de nieuwe taakrouter gebruikt.

| Toets, vijf kinderen per arm | G3-recombined | G2 op dezelfde seeds | p tegen G2 | p tegen willekeurige router |
|---|---:|---:|---:|---:|
| Eerste set | 0,9458 | 0,8115 | 0,00397 | 0,01587 |
| Onafhankelijke herhaling | 0,9468 | 0,8120 | 0,00397 | 0,02778 |

Dezelfde vijf kandidaatgewichten en routes werden in beide proeven gebruikt.
Alle vier oorspronkelijke G1-taken blijven op de exacte G2-expert van het
betreffende kind. Een opgeslagen G3-kind bevat vijf complete bronnetwerken;
de [CPU-proef](G3_RUNTIME_BENCHMARK_20261009.md) verifieerde alle 25
bronhashes en mat bij DoorKey-8 0,0580 ms per toestand tegenover 0,0560 ms
voor G2. Het gewichtbestand is ongeveer 5,03 keer zo groot. UnlockPickup
bleef in beide geheime toetsen onopgelost; CPU-energie is niet gemeten.

## Gereed voor een menselijke review

`scripts/import_g3_successor.py check` is alleen-lezen. Het verifieert de
hashes van de twee protocollen en rapporten, de onafhankelijke seedset-ID's,
alle geregistreerde promotievoorwaarden, de routes, de kandidaatgewichten in
beide proefregistries en de exacte brongewichten. Het verifieert ook de huidige
hash-chain van de officiële lineage. De geheime seeds en salt worden niet
geopend of gepubliceerd.

```bash
PYTHONPATH=. python3 scripts/import_g3_successor.py check
```

Op de huidige host geeft deze controle `ready_for_review: true`,
`already_imported: false` en review-SHA-256
`99b98ac8238654fc9cb3621eee1592602d1fc91575b5c6e479ce5b9dd2a62755`.
Dat is **geen** menselijke goedkeuring.

Een reviewer die de scores, beperkingen en 5,03× opslagkosten accepteert,
legt een afzonderlijk `approval.json` vast met exact deze sleutels:

```json
{
  "candidate_generation": "G3-recombined",
  "decision": "approve",
  "review_sha256": "99b98ac8238654fc9cb3621eee1592602d1fc91575b5c6e479ce5b9dd2a62755",
  "reviewed_by": "<identiteit in allowed_signers>",
  "approved_at": "<ISO-8601-tijdstip met tijdzone>"
}
```

De reviewer tekent de **exacte bytes** van dat bestand met een eigen SSH-sleutel
en de namespace `toddler-lineage`:

```bash
ssh-keygen -Y sign -f /pad/naar/reviewersleutel -n toddler-lineage /pad/naar/approval.json
PYTHONPATH=. python3 scripts/import_g3_successor.py apply \
  --approval /pad/naar/approval.json \
  --signature /pad/naar/approval.json.sig \
  --allowed-signers /pad/naar/allowed_signers
```

De `allowed_signers`-regel koppelt de identiteit in `reviewed_by` aan de
vertrouwde publieke sleutel. `apply` weigert een gewijzigde review, een
ongeldige handtekening of een ander bestaand G3-model. Na verificatie kopieert
het de vijf bevroren kinderen atomair naar de officiële registry, schrijft hun
ouders in de hash-chained lineage en registreert precies één verdict
`survived`. Bij een onderbreking kan dezelfde ondertekende import hervat worden;
een andere selectie wordt geweigerd. Het ongetekende bronreviewpakket in Git
blijft bevroren; de ondertekende goedkeuring blijft bij de beheerder.

## Dashboard

`scripts/build_dashboard_data.py` toont G3 vóór import als `wacht op review`.
De G3-taakpunten zijn alleen geaggregeerde geheime scores; er worden geen
per-kind-taakscores verzonnen. De kaart vergelijkt G3 met G2 **binnen dezelfde
seedset**. De openbare G2-punten in de historische grafiek zijn geen eerlijke
directe controlegroep voor de geheime G3-punten. Na een geldige import wordt
het verdict uit de officiële lineage gelezen en toont het dashboard G3 als
`overleefd`.
