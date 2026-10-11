# Preschooler: uitvoeringsprotocol voor de eerste modelfamilie

**Status 11 oktober 2026:** voorbereiding. De eerste van tien modelkandidaten, Qwen3-4B-Instruct, heeft één volledig private VirtualV-tekstprofiel doorlopen; de overige negen en de mix zijn nog niet gelijkwaardig vergeleken. Er is nog geen geaccepteerd trainingscorpus, getrainde Preschooler-checkpoint of private promotiescore. Het [businessconcept en de 32B-begroting](../businessplan/PRESCHOOLER_4B_BUSINESS_CONCEPT.md) bevatten commerciële hypothesen, geen gerealiseerde resultaten.

## Afbakening en acht leergangen

Toddler blijft het samengestelde leer- en agentsysteem. `Preschooler-4B-v0.1` leert uit [tien vastgepinde lokale 3–4B-kandidaten](preschooler-teacher-roster.json). De studentbasis wordt na identieke VirtualV-metingen gekozen; de eerste trainbare shortlist is Qwen3-4B-Instruct, Phi-4-mini-Instruct en SmolLM3-3B. Het bestaande Qwen-recept is een voorlopige default, geen uitspraak dat Qwen de beste basis is. Acht domeinen worden apart getraind en getoetst:

| ID | Leergang | Objectieve kernproef | Afzonderlijke begripstoets |
| --- | --- | --- | --- |
| `mathematics_logic` | Rekenen, logica, puzzels | numerieke oplossing / formele check | nieuw probleemtype en uitleg |
| `physics` | Natuurkunde | eenheden, simulatie of bekende oplossing | transfer naar gewijzigde randvoorwaarden |
| `chemistry` | Scheikunde | massabalans, structuur- of simulatiecheck | mechanisme op nieuwe stofgroep |
| `software` | Architectuur, code, debug | tests en onafhankelijke review | nieuwe repository en foutfamilie |
| `financial_data` | Data, signalen en causale timing | tijdsplit en reconciliatie | onbekende dataset zonder leakage |
| `language_culture` | Engels, Nederlands, geschreven Kantonees, geschiedenis en cultuur | brongebonden feiten en taalregels | vertaling en context in nieuwe bron |
| `gamedev_ui` | Gamedev en toegestane GUI-taken | uitvoerbare game-/UI-test | onbekende spelmechaniek |
| `agent_planning` | Planning, verificatie en gedrag | verifieerbare taakvolgorde | foutanalyse en herstelplan |

Elke leergang bevat gelabelde voorbeelden van **kennis**, **begrip/transfer** en **vakhandeling**. Deze labels zijn evaluatiedimensies; een hoog gemiddeld cijfer bewijst geen academisch deskundige zonder onafhankelijke nieuwe taken en deskundige beoordeling. Fysieke bediening en consumentengames zijn aparte veiligheids- en toestemmingspoorten.

## Data-ingang en controle

`scripts/prepare_preschooler_training.py` neemt JSONL-records met schema `preschooler-evidence/v2`. Per item zijn nodig: eigen taak-ID en domein; train/development-split; bronfamilie, licentie en bronhash; prompt; minstens twee kandidaten uit verschillende modelfamilies met elk model-ID, gewichtshash, licentie en expliciet toegestaan gebruik van de output voor training; geselecteerd of geverifieerd samengesteld antwoord met bijdrage-ID's; onafhankelijke objectieve of deskundige verificatie met bewijshash; en geslaagde bron-/semantische overlapcontrole. De standaardexport vereist tien unieke lokale leraar-ID's over de hele batch, niet tien oproepen per item. Een pilot mag `--min-teacher-models 2` gebruiken. Het oudere v1-schema blijft leesbaar voor bestaande, apart gelabelde bewijssets. Een afzonderlijke beheerder levert `preschooler-protected-commit/v1` met verzegelde prompt-hashes en holdoutcommitment. De converter weigert cloudoutput, niet-toegestane bronnen of modeloutputs, zelfreview, exacte overlap, gedupliceerde vragen en een bronfamilie die beide splits raakt.

Een voorbeeld met tien model-ID's bewijst geen tien onafhankelijke perspectieven. Roteer op de ontwikkelset vier complementaire kandidaten per vraag en laat alle tien modellen op 5% van de nieuwe vragen antwoorden. Meet vervolgens foutcorrelatie, aanvullend orakelplafond, afwijzingspercentage, latency en energie. Alleen een objectieve check of onafhankelijke expert kiest/synthetiseert het trainingsantwoord. Bewaar alle ruwe kandidaten buiten Git; de SFT-export bevat alleen de gekozen tekst en cryptografische hashes/herkomst van de kandidaten.

De roster staat nu op `output_training_rights: pending_per_model_review`. De standaard-CLI weigert een v2-export totdat per gebruikt model de outputrechten na broncontrole expliciet op `approved` staan en de evidence exact de gepinde revisie, modelfamilie en licentie noemt. Downloaden en benchmarken vereisen deze trainingsgoedkeuring niet.

```bash
PYTHONPATH=. python3 scripts/prepare_preschooler_training.py \
  --evidence /private/preschooler/verified-local.jsonl \
  --protected-commitment /private/preschooler/protected-commit.json \
  --out /private/preschooler/sft-v2
PYTHONPATH=. python3 scripts/train_preschooler_4b.py \
  --data /private/preschooler/sft-v2 --domain software \
  --base-alias qwen3-4b-instruct
```

De tweede opdracht is een **preflight**: controle van dataset, hashes en splits, zonder download of GPU-werk. Dezelfde opdracht accepteert als basisalias ook `phi4-mini-instruct` en `smollm3-3b`; vergelijk gelijke QLoRA-canaries vóór keuze. Voor een werkelijk trainingsproces is `pip install '.[preschooler]'` nodig en moet een GPU expliciet zijn geleased. Daarna `CUDA_VISIBLE_DEVICES=<ada-index> ... --train --out /private/preschooler/runs/software-v1`. `--max-steps N` beperkt een canary; standaard zijn het twee epochs. Het script schrijft alleen kandidaatadapters en `run.json` met `promotion_eligible: false`. Model- en datasetbestanden blijven buiten de publieke repository. De echte Tensor/LoRA-stack, kernelkeuze, batch en energie moeten op een kleine canary worden bevestigd vóór de acht lange runs.

De converter kan **geen** semantische leakage onafhankelijk vaststellen. Het opgegeven overlaprapport is daarom een vereiste van de onafhankelijke evaluator, niet een vrijbrief om een willekeurige `passed`-waarde te vertrouwen. De eindtoets mag nooit door de dataverzamelaar of promptoptimizer worden gelezen.

## Meet- en stopregels

1. Benchmark alle tien leraarcheckpoints eerst apart op dezelfde private VirtualV-packs, met hun eigen loadercanary, 256-token-t/s, GPU-board-Wh en llmwatch-systeemspoor. Bereken daarna voor elk technisch zinvol subset de beste vaste route, willekeurige route en het onhaalbare per-item-orakel. De bestaande drie-modelproef heeft slechts 4,44 procentpunt optimistische ruimte boven het beste losse model; dit is geen private bewijswaarde. Alleen subsets met complementaire fouten krijgen een live router-, cascade- of verifierproef.
2. Verzamel voor elke leergang eigen, rechten-gedekte en onafhankelijk geverifieerde voorbeelden. Begin met 10.000 voorbeelden per domein; steekproef en begroting komen uit `scripts/estimate_academic_schooler.py`.
3. Train acht onafhankelijke adapters op een bevroren 4B-basis; vergelijk met ongetrainde basis, een simpele vaste modelroute en de bevroren Toddler + Teacher + MoM-configuratie. Volg oude vaardigheden na elke leergang. Consolideer pas als gezamenlijke transfer en retentie aantoonbaar haalbaar zijn.
4. Gebruik de actieve VirtualV anti-contaminatie- en private-specialistenprotocollen. Registreer prompts, modelrevisie, decodeerinstellingen, 256-token-t/s, GPU-board-Wh en apart gemeten volledige systeem-kWh. Beoordeel Claude Haiku 5.5 op minstens twee vastgelegde effort-niveaus (`low` en `medium`) met drie herhalingen per toegestane, niet-gevoelige toetsvraag; geef geen `temperature`, `top_p` of `top_k` mee. [Anthropic beschrijft het standaardniveau en de parameterbeperkingen](https://platform.claude.com/docs/en/models/haiku-5-5/overview). Cloudantwoorden voeden geen training.
5. Voor modelpromotie: vooraf gepowerde gepaarde toets, positieve ondergrens van het 95%-interval op de hoofdscore, geen domeinregressie groter dan vijf procentpunt en nul kritieke incidenten. Systeem-G5 houdt afzonderlijke navigatie-, cognitie-, software- en fysieke poorten. Onbesliste of mislukte kandidaten blijven buiten productie.

Componentattributie kan met `scripts/analyze_component_attribution.py` uit een vooraf gerandomiseerde, taakidentieke `toddler-component-arms/v1`-proef worden berekend. De benodigde armen zijn `baseline`, `full`, `plus/<component>` en `minus/<component>` voor iedere toepasselijke component. Iedere rij vermeldt dezelfde profielhash en taak-ID, een onafhankelijke acceptatie, latentie, GPU-board-Wh en systeem-kWh. Het resultaat is analyse, nooit automatisch promotie.

## Academic Schooler-32B en publicatie

Een 32B-fase begint op een afzonderlijke Qwen3-32B-basis (`9216db5781bf21249d130ec9da846c4624c16137`). De acht 4B-curricula leveren datasetontwerp en foutanalyses, geen gewichtsexpansie. Een 8× H100 80GB-LoRA-proef en 4 TB vrije NVMe-werkruimte zijn de budgetreferentie; de berekening is reproduceerbaar met `scripts/estimate_academic_schooler.py`. Gezien de huidige bezetting van EDS1/EDS2 en de GPU's start er geen grootschalige run zonder opslag- en GPU-leasecontrole.

Publiceer pas na onafhankelijke bewijscontrole twee onderscheiden artefacten: (1) Toddler als systeem met configuratie, adapters en evaluatiecode; (2) de werkelijk getrainde Preschooler-checkpoint als eigen Hugging Face-modelrepo met `base_model`, licenties, dataherkomst, trainprofiel, niet-geslaagde proeven en beperkingen. Een publieke demonstratie mag de verzegelde toetsset niet tonen. De huidige repository bevat **geen** Preschooler-gewichten en doet geen 32B-kwaliteitsclaim.
