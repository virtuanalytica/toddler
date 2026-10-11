# Doorlopende kennisopbouw voor 10.000 vakbekwaamheden

**Start 11 oktober 2026.** De atlas heeft 10.000 toetsbare specificaties, verdeeld over 100 onderwerpen. Deze pipeline laat een dagelijkse cronjob alle ID's bezoeken en maakt apart zichtbaar wat nog een primaire bron, een les of een onafhankelijke review nodig heeft. Een bezoek of kandidaatbron is geen inhoudelijke voltooiing.

## Vier poorten

| Fase | Bewijs | Automatisch? |
|---|---|---|
| Bron ontbreekt | Onderwerp staat in de researchwachtrij | Ja, OpenAlex levert slechts *kandidaten* |
| Bronbundel gecontroleerd | `knowledge/topic_sources.jsonl` met primaire URL, uitgever, rol en controledatum | Nee, relevantie en broninhoud worden gecontroleerd |
| Lesconcept | `lesson.md`, oefening, verwachte output, faalcontrole en bron-URL's in `capability.json` | Een agent kan een concept schrijven; status blijft `lesson_draft` |
| Beoordeelde les | Onafhankelijke vakreview met bron-, veiligheids- en reproduceerbaarheidscheck | Nee; pas dan `reviewed_lesson` met reviewbewijs |

Een afzonderlijke geheime promotietoets bepaalt of een *specifieke Toddler* de vakbekwaamheid beheerst. De publieke wachtrij bevat geen antwoorden, private seeds of toetsprompts. Teacher mag lesconcepten verkennend aanbieden, maar ze niet als gevalideerde labels of promotiebewijs gebruiken.

## Dagritme en capaciteit

Bij 64 ID's per dag is een eerste volledige ronde in `ceil(10000/64) = 157` dagen bereikbaar. Zes maanden bieden daarmee circa 23 dagen speling voor uitval. **Dat is alleen een dekking van de wachtrij.** Voor 10.000 daadwerkelijk beoordeelde lessen is gemiddeld circa 56 afgeronde reviews per dag nodig over 180 dagen. De huidige capaciteit of reviewkwaliteit bewijst dat tempo niet; het auditrapport toont de feitelijke achterstand. Bronnen worden per onderwerp gedeeld, zodat bronresearch maximaal 100 bundels vraagt, terwijl praktijkopdracht en review per vakbekwaamheid afzonderlijk blijven.

Een dagtick is eenmalig en idempotent, neemt een niet-blokkerende bestandslock, schrijft een SHA-256-gecontroleerde JSONL-wachtrij en voegt een gehashte gebeurtenis toe aan een append-only ledger. Een gemiste dag verschuift geen teller; de volgende dag gaat het werk bij de eerstvolgende ID verder. Na ID 10.000 begint een nieuwe ronde, zodat resterende blokkades opnieuw worden getoond. Bronvoorstellen van OpenAlex zijn metadata en krijgen nooit automatisch lesstatus.

## Commando's

```bash
python3 scripts/knowledge_pipeline.py audit
python3 scripts/knowledge_pipeline.py discover --max-topics 2
python3 scripts/knowledge_pipeline.py tick --daily-target 64
```

Standaard staat de runtime buiten Git op `/media/knight2/claude-data/knight1/knowledge/capability-atlas/pipeline`. Gebruik `--state-dir` voor een proef. De wachtrij in `queues/YYYY-MM-DD.jsonl` bevat het ID, de vakbekwaamheid, de bronbundel en de vereiste volgende actie. `source_candidates/<domein>/<onderwerp>.json` is een **onbeoordeelde** literatuurlijst uit de [OpenAlex Works API](https://help.openalex.org/api/). Voeg na inhoudelijke verificatie alleen toegestane links toe aan de versiebeheerde bronregistratie; kopieer geen volledige papers of datasets zonder licentiecontrole.

## Cronvoorstel

Na installatie op een stabiele checkout met dezelfde gebruiker:

```cron
SHELL=/bin/bash
TZ=Europe/Amsterdam
5 2 * * * cd /media/knight2/EDS2/projects/toddler-knowledge-production && /usr/bin/python3 scripts/knowledge_pipeline.py discover --max-topics 2 >> /media/knight2/claude-data/knight1/knowledge/capability-atlas/pipeline/discover.log 2>&1
20 2 * * * cd /media/knight2/EDS2/projects/toddler-knowledge-production && /usr/bin/python3 scripts/knowledge_pipeline.py tick --daily-target 64 >> /media/knight2/claude-data/knight1/knowledge/capability-atlas/pipeline/tick.log 2>&1
35 2 * * * cd /media/knight2/EDS2/projects/toddler-knowledge-production && /usr/bin/python3 scripts/knowledge_pipeline.py audit >> /media/knight2/claude-data/knight1/knowledge/capability-atlas/pipeline/audit.log 2>&1
```

De uitvoerende checkout moet vastgepind worden op een beoordeelde release; update die bewust, niet midden in een tick. Monitor de exitcodes en `inventory.json`. De huidige zes bronbundels zijn door de assistent tegen primaire publicaties gecontroleerd; vakinhoudelijke menselijke review is nog open. De registratie noemt alle bronnen als links, zonder auteursrechtelijk beschermde volledige tekst in Git.
