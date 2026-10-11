# Signals: twee targethorizons als onderzoeksmodel

Status: 11 oktober 2026, **alleen ontwikkelproef**. De competitie-specialist uit [FINANCE_SPECIALIST_AGENTS.md](FINANCE_SPECIALIST_AGENTS.md) gebruikt de officiële Signals v3.0-data. Deze proef traint twee LightGBM-modellen op dezelfde vooraf vastgelegde features, één voor `target_jupiter_60` en één voor `target_jupiter_20`. Per week worden hun voorspellingen afzonderlijk gerangschikt en met een **vooraf gekozen 50/50-verhouding** gemiddeld. De evaluatietarget blijft Jupiter-60.

| Variant | Gemiddelde wekelijkse Spearman op Jupiter-60, 2023–2024 | Block-bootstrap 95%-interval |
|---|---:|---:|
| Jupiter-60 alleen | 0,02582 | 0,02158–0,03315 |
| Jupiter-20 alleen | 0,02680 | 0,02277–0,03405 |
| Mix 50/50 | **0,02753** | 0,02341–0,03509 |
| Mix met per week geschudde Jupiter-20 voorspelling | 0,01895 | 0,01508–0,02402 |

Het gepaarde verschil tussen de mix en Jupiter-60 alleen is **+0,00171**, met een block-bootstrap-interval van **+0,00059 tot +0,00305** over 91 weken. Beide modellen zijn getraind op 5.886.986 rijen tot en met 2022; de ontwikkelset bevat 598.994 rijen uit 2023–2024. De targethorizons overlappen, zodat de block-bootstrap op negen opeenvolgende weken werkt. De bron- en modelhashes staan in `signals_multihorizon_v1/experiment.json` op claude-data.

Dit is een historische rangcorrelatieproxy, geen officiële Signals Alpha, MPC, payout of bewijs voor het 95e percentiel. Het ontwikkelresultaat heeft de keuze voor deze variant beïnvloed; het betrouwbaarheidsinterval is daarom **geen onafhankelijke bevestiging van generalisatie**. De 2025+ holdout was al voor een eerdere baseline geopend en wordt niet opnieuw als nieuwe holdout geclaimd. Promotie vereist prospectieve volwassen targets, officiële rondescores en een vergelijking met de vooraf vastgezette Jupiter-60 baseline.

Op de officiële livekopie van **9 oktober** is één lokale voorspelling voor **7.177 tickers** vastgelegd. Het manifest noemt SHA-256 van bron, beide modellen en voorspelling; `submitted=false` en `stake=0`. De uitkomst is nog niet meetbaar omdat de 60-daagse target moet rijpen. Het script kan dezelfde forecast zonder overschrijven opnieuw herkennen. Het maakt geen competitie-inzending.

De bestaande `scripts.finance_daily_cycle` voert deze extra lokale voorspelling na een geslaagde officiële synchronisatie uit wanneer het getrainde `experiment.json` aanwezig is. Daardoor blijft de onderzoeksvariant op een nieuwe machine zonder modelbestand optioneel; op deze machine worden toekomstige voorspellingen met dezelfde bevroren modellen vastgelegd. De dagrapportage noemt de extra stap en geeft een foutstatus als die faalt.

## Reproductie

```bash
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 python3 -m scripts.research_signals_multihorizon train
python3 -m scripts.research_signals_multihorizon predict
pytest -q tests/test_signals_multihorizon.py
```

De standaard bronmap is `/media/knight2/claude-data/knight1/finance-agent-data/official` en de artifactmap is `/media/knight2/claude-data/knight1/finance-agent-runs/signals_multihorizon_v1`; beide zijn met CLI-opties te vervangen. Het script materialiseert bij training alleen datums vóór 2025 en weigert ontbrekende live tickers, dubbele tickers, gewijzigde modelhashes en ongeldige voorspellingen.
