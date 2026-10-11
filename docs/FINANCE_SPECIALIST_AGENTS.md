# Financiële agents: eerste reproduceerbare trainingsronde

Status: **onderzoek**, 11 oktober 2026. Vier afgebakende rollen delen een manifest, hashes en een tijdgescheiden experiment. De code doet geen inzending of NMR-stake. Een historisch signaal is geen bewijs voor het 95e percentiel van de live competitie.

## Rollen en bronnen

| Agent | Invoer | Uitvoer | Poort |
|---|---|---|---|
| Data-updater | Geregistreerde officiële Crypto v2.0 en Signals v3.0 Parquet, bestaande lokale collectors | Hash, schema, rijaantal en peildatum per bron | Ontbrekende bestanden of doelvelden stoppen de run; verversing volgt de bestaande Numerai-cron |
| Data-preparer | Alleen `feature_*`, datum en gepinde target | Train/dev/holdout Parquet en manifest | Doelrijen zonder volwassen target weg; gaten van > doelhorizon tussen perioden |
| Data-interpreter | Geprepareerde features | Vooraf bepaalde, rijgewijze momentum/volatiliteit-verschillen en featurelijst | Geen target, toekomstdata of testscore bij featurecreatie |
| Competitie-specialist | Train/dev en twee kandidaatmodellen | Ridge/LightGBM, ontwikkelscore, bevroren keuze; afzonderlijke eenmalige holdout | Live score/percentiel en inzet blijven buiten automatische promotie |

Crypto gebruikt `crypto/v2.0/train.parquet` met `target_binned_return_20` zoals dat bestand hier daadwerkelijk heet. De huidige [Numerai Crypto-scoring](https://docs.numer.ai/numerai-crypto/scoring/definitions) noemt CORR en MMC over 20D2L; onze gemiddelde dagelijkse Spearman is slechts een snelle *proxy* en bevat geen MMC. De [Crypto-datapagina](https://docs.numer.ai/numerai-crypto/data) noemt elders 30 dagen en wijkt ook in bestandsnamen af van de actuele API-lijst. Daarom ligt de concrete dataset en targetnaam in het manifest vast; vóór live gebruik verifiëren we de rondedefinitie opnieuw.

Signals gebruikt de officiële v3.0 train- en validationbestanden, `target_jupiter_60` en de numerieke starterfeatures. De [v3.0-documentatie](https://docs.numer.ai/numerai-signals/data) bevestigt dat recente 60-daagse targets nog leeg kunnen zijn; die rijen worden verwijderd. De historische Spearman is **geen** [Signals Alpha of MPC](https://docs.numer.ai/numerai-signals/scoring), laat staan de volledige live score. De 300-koloms neutralizer en een eigen point-in-time aandelenfeed zijn zinvolle volgende datasets voor een eerlijkere Signals-proef; de eerste ronde gebruikt ze nog niet. `feature_country` vraagt expliciete categorische codering en blijft in deze eerste run weg.

De huidige lokale officiële datasets worden als bron gebruikt; de nieuwe trainingsartifacts staan op claude-data. De bestaande 03:30 Numerai-verversing was op 11 oktober nog bezig met de vorige cyclus en hield de lock vast; op EDS2 was circa 13 GB vrij terwijl het script minstens 25 GB eist. De updater meldt die toestand en omzeilt de opslaggrens niet. De vijf officiële Crypto/Signals train-, validatie- en livebestanden zijn op claude-data gespiegeld met een reserve van 200 GiB. De lokale Signals-trainkopie en de vers gedownloade kopie hebben verschillende bestandshashes door metadata; alle tabelwaarden zijn gelijk bevonden.

## Modelkeuze en anti-contaminatie

- Ridge is een eenvoudige referentie. LightGBM is de eerste niet-lineaire tabulaire kandidaat. Een zwaardere generatieve LLM is geen logische primaire prijsvoorspeller voor deze kleine tabulaire feature-set.
- Een lokaal LLM mag hypothesen over causaliteit, bronnen en featurefouten beoordelen op basis van **alleen schema en ontwikkelresultaten**. Geen holdoutrijen, holdoutscore, geheime toetsitems of live private data gaan in zo'n prompt. LLM-voorstellen worden niet automatisch features of labels.
- De eerste review gebruikt de bestaande lokale `mom-live` endpoint (Qwen3.6-35B-A3B-NVFP4). Het antwoord bevatte ook een niet-bestaande `feature_close_ema_20d` en is daarom terecht alleen als onbeoordeelde hypotheselijst opgeslagen. Voor voorspelling gebruiken we Ridge en LightGBM; er is geen nieuw groot LLM nodig of gedownload.
- De regels voor de tijdgrenzen en featuretransformaties staan in Git vóór de holdoutopening. De `evaluate-holdout` stap schrijft een eenmalig resultaat en weigert heropening in dezelfde artifactmap. Wie daarna iteratief wijzigt, moet een nieuwe prospectieve periode reserveren.
- Positieve onderzoeksresultaten vereisen nog round-aware CORR/MMC en voor Signals Alpha/MPC, neutralisatie, spreiding over regimes, transactie- en API-kosten, een shuffled-target control, point-in-time herkomst, live inzendingen zonder stake en ten slotte de actuele leaderboardverdeling. De 95e percentielclaim volgt uitsluitend uit die live verdeling.

## Reproductie

```bash
python3 -m pip install -e '.[finance]'
python3 -m toddler.finance_agents inventory crypto
python3 -m toddler.finance_agents refresh-status crypto
python3 -m toddler.finance_agents sync-official crypto
python3 -m toddler.finance_agents prepare crypto
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 python3 -m toddler.finance_agents train crypto --threads 8
python3 -m toddler.finance_agents score-proxies crypto
python3 -m toddler.finance_agents llm-review crypto
python3 -m toddler.finance_agents leaderboard crypto
# Pas na bevriezing van de kandidaat en het protocol:
python3 -m toddler.finance_agents evaluate-holdout crypto
# Een lokale vooruitblik zonder inzending of stake:
python3 -m toddler.finance_agents predict-live crypto
```

Vervang `crypto` door `signals` voor de andere competitie. De standaard bronmap is `/media/knight2/EDS2/projects/numerai-signals/data`; de artifactmap is `/media/knight2/claude-data/knight1/finance-agent-runs`. Gebruik `--data-root` en `--out` om expliciet andere locaties te kiezen. Leg voor elke latere run het Git-commit, bronhashes, datasetversie, modelhashes en UTC-datum vast. Een hoog ontwikkelresultaat is een selectiecriterium, geen claim over toekomstig rendement.

## Eerste ontwikkeluitkomsten

| Competitie | Train/dev rijen | Ridge Spearman | LightGBM Spearman | Gekozen | Canonical CORR-transformatie, gekozen model | Geschudde-targetcontrole |
|---|---:|---:|---:|---|---:|---:|
| Crypto v2.0 | 385.396 / 65.400 | 0,1645 | 0,1561 | Ridge | 0,1491 | 0,0028 |
| Signals v3.0 | 5.886.986 / 598.994 | 0,0104 | 0,0258 | LightGBM | 0,0236 | 0,0002 |

De intervallen zijn block-bootstrap-intervallen over dagen/weken vanwege overlappende targethorizons. Crypto Ridge: 0,1109–0,2215; Signals LightGBM: 0,0216–0,0331. `numerai_tools.scoring.numerai_corr` levert een tweede, meer wedstrijdachtige historische proxy, maar geen officiële payoutmeting. Volledige machineleesbare resultaten, inclusief bron- en modelhashes, liggen buiten Git in de artifactmap.

Na de codefreeze is de historische holdout op 11 oktober **eenmalig** geopend. Crypto Ridge: 0,0936 gemiddelde dagelijkse Spearman over 138 dagen, block-bootstrap 95%-interval 0,0355–0,1518. Signals LightGBM: 0,0084 over 66 weken, interval 0,0039–0,0135. Beide vielen terug ten opzichte van de ontwikkelperiode. De behaalde 95e percentiel is **niet aangetoond**; volgende modelwijzigingen kunnen alleen met nieuwe toekomstige rondes prospectief worden gepromoveerd. Voor 9 oktober zijn lokale voorspellingen geschreven voor 300 Crypto-tokens en 7.177 Signals-aandelen, zonder inzending of stake. Ze wachten op volwassen doelvariabelen.

De read-only publieke accountlijst op 11 oktober 02:04 UTC zet `develuse` bij Crypto op rang 77/278 (72,66e percentiel) en Signals op 143/492 (71,14e percentiel). Voor de top 5% zijn de huidige grensrangen 14 en 25. Dit is de positie van de bestaande gestakete accountportefeuille, **niet** de score van de nieuwe onderzoeksagent. De actuele competitiepositie wordt bij iedere meting opnieuw vastgelegd; een vaste CORR-drempel is geen vervanging voor de leaderboardmeting.
