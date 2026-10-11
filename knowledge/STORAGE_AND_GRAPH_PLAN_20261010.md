# Opslag en kennisgraaf voor 10.000 Toddler-capabilities

**Peildatum 10 oktober 2026.** De atlas heeft 10.000 capability-adressen, 100 onderwerpen en tien uitgewerkte lesconcepten. De cijfers hieronder zijn capaciteitsramingen, geen reeds gedownloade collectie. Reken in decimale GB/TB. [De herberekenbare begroting](storage_budget_20261010.json) komt uit `scripts/capability_storage_budget.py`.

**Aanvulling 11 oktober 2026:** alle 10.000 adressen hebben nu een unieke, gegenereerde [vakbekwaamheidsspecificatie](COMPETENCY_CATALOG.md) met product, maat, faalwijze, bewijs en rubric. Het aantal brongebonden lesconcepten blijft tien. De opslagraming voor papers, video en oefeningen verandert daardoor niet; de nieuwe teksten zijn kleine Git-bestanden, geen 10.000 verzamelde bronbundels.

**Tweede aanvulling 11 oktober 2026:** zes extra brongebonden lesconcepten en [zes onderwerp-bundels](topic_sources.jsonl) brengen het totaal op **16**. De t-SNE-export is opnieuw berekend op 16 teksten (`perplexity=5`, `trustworthiness@3=0,907197`); dit blijft een verkennende kaart. De LightRAG-*payload* telt nu 16 chunks, 53 entiteiten en 57 relaties. De eerdere ingerichte runtime met 10 chunks is een historische pilot en is nog niet opnieuw geïndexeerd. De [dagelijkse kenniswachtrij](../docs/KNOWLEDGE_COLLECTION_PIPELINE.md) houdt bronresearch, lesontwikkeling en review afzonderlijk bij.

## Inhoud en ruimte per map

| Onderdeel | Planningsband per echt uitgewerkte capability | Fysieke indeling |
|---|---:|---|
| Eigen theorie, instructies, rubrics, transcripties | 1–20 MB | Versiebeheerde tekst en hash |
| 5–20 papers, gemiddeld 2–10 MB | 10–200 MB logisch | Eén toegestane bronkopie per paper, gedeeld via blobhash |
| 5–30 website-snapshots à 0,1–2 MB | 0,5–60 MB logisch | URL, revisie, licentie, datum en optionele snapshot |
| 1–5 relevante code-repositories | 0,05–10 GB logisch | Eén gespiegelde repository per commit; capability-map bevat een verwijzing, geen clone |
| Oefeningen, datasets, voorbeeldresultaten | 20 MB–2 GB | Dataset per onderwerp delen; unieke opdrachtuitvoer apart bewaren |
| 2–10 video's van 10–30 minuten bij 2–8 Mbit/s | circa 0,3–18 GB | Eén geautoriseerd videobestand plus transcript; anders alleen een bronlink |
| LightRAG-, embedding- en codegraph-index | Aanvankelijk enkele MB per bronbundel | Eén index per corpus/modelversie of repositorycommit, nooit 10.000 kopieën |

Een video van 20 minuten bij 3 Mbit/s kost ongeveer **450 MB**; tien video's zijn circa **4,5 GB**. De werkelijke opslag hangt sterk af van resolutie, compressie, toegestane lokale kopieën en gedeelde bronnen. PhD-niveau komt uit kwaliteit van bewijs en oefeningen; de hoeveelheid bytes is geen maat daarvoor.

De huidige 10.000 mappen zijn 100 onderwerpen × 100 werkwijze/situatiecombinaties. Als alle zware media per map worden gekopieerd, is dat economisch onzinnig. De drie scenario's nemen daarom per onderwerp een gedeelde bronbundel en per capability unieke oefen- en bewijsdata aan:

| Scenario | Onderwerp-bundel | Uniek per map | Indexen totaal | Logisch per map / 10.000 | Fysiek primair | Met 20% vrije werkruimte | Twee volledige kopieën |
|---|---:|---:|---:|---:|---:|---:|---:|
| Lean | 2 GB | 20 MB | 20 GB | 2,022 GB / 20,22 TB | **0,42 TB** | 0,525 TB | 0,84 TB |
| Balanced | 8 GB | 100 MB | 50 GB | 8,105 GB / 81,05 TB | **1,85 TB** | 2,312 TB | 3,70 TB |
| Rich | 20 GB | 500 MB | 200 GB | 20,52 GB / 205,2 TB | **7,20 TB** | 9,00 TB | 14,40 TB |

Voor 10.000 werkelijk **verschillende** onderwerpen vervalt de sterke onderwerpdeduplicatie; de logische totalen komen dan dichter bij de fysieke eis. Een onafhankelijke tweede kopie moet op een andere foutdomeinlocatie staan. De twee-kopie-kolom is niet een extra reserve boven op de werkruimtekolom.

Op deze host was vrij: EDS1 178 GB, EDS2 62,7 GB, claude-data 1.852 GB. Met de eerder gekozen minimumreserve van **200 GB per SSD** geven EDS1 en EDS2 nu geen nieuwe bulkruimte, en claude-data maximaal 1.652 GB. Lean past als primaire collectie; balanced past al zonder werkruimte of tweede kopie niet volledig op claude-data. Sla daarom de kleine catalogus, manifesten en code in Git op, en bouw de gedeelde blobstore en pilot-index in `/media/knight2/claude-data/knight1/knowledge/capability-atlas/`. Download geen 10.000 repositoryklonen of video's bij voorbaat.

## Samenhang: drie verschillende grafen

1. **Canonieke leerrelaties.** `CAPABILITY_INDEX.txt` blijft het ID-register. De eerste tien lessen hebben een [acyclische, redactionele prerequisite-graaf](graph/prerequisites.json). Voeg per beoordeelde les expliciete `requires`, `uses`, `compares_with` en `assessed_by`-relaties met bron en reviewer toe. De beoordeelde graaf bepaalt later de leerroute. Dit is de bron van waarheid voor logische samenhang.
2. **t-SNE-kaart.** `scripts/capability_tsne.py` projecteert uitsluitend de tien bestaande lessen uit TF-IDF naar 2D. [De pilotuitvoer](graph/lesson_tsne.json) heeft een vaste seed, perplexity 3 en `trustworthiness@3 = 0,853333`. Dichtstbijzijnde lessen komen uit de oorspronkelijke cosine-similariteit, niet uit 2D-afstand. Bij tien teksten is de kaart verkennend; t-SNE bepaalt geen curriculum, afhankelijkheid of kwaliteit. De [scikit-learn-documentatie](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) beschrijft de stochastische, niet-convexe projectie.
3. **Bron- en codegraaf.** `scripts/capability_graph_links.py` verbindt alle 10.000 IDs met een domeinbrede kandidaat-repository en een *bestaande, hash-gecontroleerde* lokale source-file-node. `graph/capability_repository_links.jsonl` labelt dit als `domain_discovery_candidate` en `related_local_context`, nooit als bewijs dat de repo die specifieke capability implementeert. Tien lesconcepten hebben daarnaast primaire bron-URL's. De [meeleverbare codegraph-context](graph/toddler_codegraph_context.json) bevat de gekoppelde bestanden, hun symbolen en directe afhankelijkheden; de volledige geïsoleerde GitNexus-compatibele codegraph omvat circa 3.500 nodes en 7.900 edges. Een repositorycommit en bronhash zijn nodig om links later opnieuw te valideren.

## LightRAG en GitNexus

De [LightRAG-pilotexport](graph/lightrag_pilot_payload.json) bevat alleen de tien inhoudelijke lessen, hun bron-URL's, repositorykandidaten en codegraph-nodes. `scripts/capability_lightrag.py ingest` indexeerde deze met een lokaal gecachte 384-dimensionale MiniLM-embedding in een geïsoleerde LightRAG 1.5.7-werkmap op claude-data: **10 chunks, 28 entiteiten, 33 relaties, 302.151 bytes**. De directe custom-KG-route gebruikt geen LLM-extractie; er is geen antwoordkwaliteit of retrievalscore gemeten. LightRAG meldt dat de standaard bestandsbackends voor kleine pilots zijn en raadt voor grote productiecollecties onder meer PostgreSQL aan; kies backend en embeddingmodel vóór bulkindexering. [LightRAG-repository](https://github.com/HKUDS/LightRAG).

De lokale codegraph is gemaakt met de reeds aanwezige `build_codebase_lightrag_gitnexus_obsidian.py`; dat is een **GitNexus-compatibele eigen index**, geen claim dat de upstream GitNexus-CLI is uitgevoerd. De upstream [GitNexus-repository](https://github.com/abhigyanpatwari/GitNexus) publiceert momenteel een [PolyForm Noncommercial-licentie](https://github.com/abhigyanpatwari/GitNexus/blob/main/LICENSE). Houd de eigen indexer als werkend pad; voor commercieel gebruik van upstream GitNexus is een passende licentie of andere implementatie nodig. Indexeer externe repositories één keer per vastgepinde commit; koppel capabilities daarna via graaf-IDs en bewaar de licentie, commit, bronhash en scope van elke verwijzing.

## Volgende kwaliteitsproef

Vul eerst 100 onderwerp-bundels met geautoriseerde primaire bronnen, transcripties en één reproduceerbare praktijkproef. Meet retrieval op verzegelde vragen: bronprecisie, correcte codeverwijzing, antwoord met bewijs, latency en opslaggroei. Pas als de tien-lessenpilot en vervolgens de 100-onderwerpenproef slagen, schaal op naar alle 10.000 adressen. Geen bron uit de privépromotietoets mag in deze kennisbank terechtkomen.
