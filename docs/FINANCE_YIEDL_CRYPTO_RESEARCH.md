# YIEDL als extra Crypto-databron: verkennende proef

**11 oktober 2026 — research-only.** [Numerai noemt YIEDL als aanvullende bron](https://docs.numer.ai/numerai-crypto/data) voor prijs/volume/momentum, sentiment en onchain gegevens. De lokale `yiedl_agg.parquet` bevat 5.519.214 rijen en drie gemiddelden per token en datum. Deze bron is niet in Git opgenomen en de historische beschikbaarheid en eventuele revisies zijn nog niet onafhankelijk gereconstrueerd. De bestaande pipeline zet `pit_safe=True` zonder bewijs per observatie; die vlag geldt hier niet als certificaat.

## Vooraf vastgelegd protocol

De officiële Crypto v2.0-rij op datum `t` krijgt uitsluitend de meest recente YIEDL-rij met bron-datum `≤ t − 2 kalenderdagen`, en hoogstens zeven dagen oud. De join gebeurt per token. Een test blokkeert kortere vertraging en controleert dat een latere bronrij niet zichtbaar wordt. Alleen 2020–2024 zijn gebruikt voor training; 2025 is de ontwikkelperiode. De 2026-rijen zijn voor deze proef niet gelezen. De eerder voor de basisagent geopende 2026-holdout kan geen onafhankelijke promotietoets voor dit tweede model zijn.

Beide LightGBM-modellen hebben dezelfde 200 bomen, 20 leaves, learning rate 0,035, acht CPU-threads en dezelfde officiële features. Het enige verschil is toevoeging van de drie YIEDL-gemiddelden. Hun bronbestand en beide modellen zijn met SHA-256 vastgelegd. Er is geen selectie op basis van live score, geen inzending en geen stake.

| 2025-ontwikkelmeting | Alleen officiële features | Officieel + YIEDL |
|---|---:|---:|
| Gemiddelde dagelijkse Spearman | 0,15609 | 0,16159 |
| Gepaard verschil | — | +0,00550 |
| Block-bootstrap 95%-interval voor verschil | — | +0,00152 tot +0,01041 |
| YIEDL-features binnen datum geschud | — | 0,15340 |

De broncoverage in 2025 is 98,8% voor PVM, 53,0% voor onchain en 98,2% voor sentiment. Het positieve verschil en de shuffled-control geven aanleiding voor een prospectieve proef; ze bewijzen geen causaal of point-in-time schoon effect. De officiële Numerai-score omvat bovendien MMC en een actuele live cohortpositie, die deze lokale rangcorrelatie niet meet.

Voor 9 oktober is een **lokale** voorspelling voor alle 300 tokens opgeslagen. YIEDL-observaties tot en met 7 oktober zijn gebruikt; de live coverage is respectievelijk 95,0%, 58,3% en 93,7%. Die voorspelling wordt pas na targetrijping en naast de bestaande basisagent beoordeeld. De bron kan worden gepromoveerd als revisie-/availability-audit slaagt en meerdere vooraf geregistreerde live rondes een stabiele gepaarde winst tonen. Daarna pas zijn een officiële CORR/MMC-vergelijking en leaderboardmeting zinvol.
Een volgende lokale forecast wordt geweigerd wanneer PVM of sentiment minder dan 80% van de live tokens dekt.

## Reproductie

```bash
python3 -m pip install -e '.[finance]'
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 python3 -m scripts.research_yiedl_crypto train
python3 -m scripts.research_yiedl_crypto predict
```

Standaardinput: de officiële spiegel op claude-data en de lokale `proofs/cross_pollination/yiedl_agg.parquet`. Uitvoer: `/media/knight2/claude-data/knight1/finance-agent-runs/yiedl_crypto_v2/experiment.json` plus gehashte modellen en een prospectief CSV-bestand. Gebruik `--official`, `--yiedl` en `--out` om de locaties te pinnen. De bron zelf wordt niet publiek gekopieerd.
