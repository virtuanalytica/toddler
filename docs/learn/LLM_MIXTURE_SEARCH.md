# LLM-modelmix als onderzoeksbaan

Naast de navigatie-experts kan Teacher per taak een **kandidaatroute** tussen
lokale LLM's zoeken. `toddler.model_mixture_search` vergelijkt gemeten
antwoordkwaliteit, tijd per antwoord, decode-tokens per seconde, GPU-board-Wh
per antwoord en benodigde gelijktijdige VRAM. Hij toont de beste losse
modelroute, de verwachtingswaarde van een willekeurige route en maximaal tien
gemengde kandidaten. De uitkomst is een onderzoeksplan, geen nieuwe Toddler,
geen automatisch serve-commando en geen promotiebewijs.
Uit de bewaarde scores per openbaar item berekent hij eerst het plafond van
een perfecte orakelrouter binnen het aantal gelijktijdige modellen en het
VRAM-budget. Een tweede plafond zonder die beperking blijft apart zichtbaar;
het haalbare plafond is nog optimistisch over latency en energie. Als het
nauwelijks boven het beste losse model ligt, heeft een echte router weinig
kans om winst te boeken.

De invoer is een JSON-export met schema `toddler-mom-public-dev/v1`.
Verplicht zijn een nieuwe openbare ontwikkelsplit, beleid
`anti_contamination_public_development`, een geslaagde overlapcontrole,
hashes van itembank, promptpack, decodeerprofiel en modelgewichten, identieke
taaknamen, minstens 30 items per model/taak en alle bovengenoemde gemeten
eenheden. De scoregewichten per taak kunnen vooraf in `task_weights` worden
vastgelegd. De zoekruimte is begrensd tot 2 miljoen routes; standaard mogen
hooguit drie modellen samen maximaal 80 GB VRAM vragen. Een rij met private
resultaten, publieke historische full-suite-scores of ontbrekende energie
wordt geweigerd. GPU-board-Wh is niet het hele systeemverbruik.

Per `models[]`-rij verwacht de export `model`, `weights_sha256`,
`access: local`, `energy_scope: gpu_board`, `resident_vram_gb` en voor elke
taak een record met `n`, `item_ids` (een geordende lijst van unieke openbare
ID's), `item_ids_sha256` (SHA-256 van compacte UTF-8 JSON van precies die
lijst), `item_scores` (één score van 0–1 in dezelfde volgorde
per openbaar item), `quality` (het gemiddelde daarvan), `latency_s`, `decode_tps` en
`gpu_board_wh_per_answer`. De hele export draagt `tasks`, optioneel
`task_weights`, `promotion_eligible: false` en
`training_overlap_check: passed`. Deze metadata zijn een controleerbare
aanleverafspraak, geen zelfstandig bewijs dat de overlapcontrole juist was.
Per taak moeten alle modellen exact dezelfde geordende ID's en hetzelfde aantal
vragen hebben; anders is zelfs een openbare modelvergelijking niet gepaard.
De willekeurige-routerondergrens gebruikt alleen losse modellen die de opgegeven
VRAM-, latency- en GPU-board-energielimieten halen.

De nachtservice verwacht deze export op
`/media/knight2/EDS2/projects/virtualv_llm/reports/toddler_mom_public_dev.json`.
Dat bestand bestaat nog niet. Totdat virtualv_llm een nieuw, vergelijkbaar
openbaar ontwikkelrapport levert, registreert Teacher
`waiting_for_public_development_export`; hij verzint geen ranglijst op basis
van de afgesloten private tabellen. Een kandidaatmix moet daarna nog op verse,
onzichtbare anti-contaminatie- en specialisttaken worden vergeleken met het
beste losse model en een willekeurige router, inclusief volledige latency en
energie per vraag.
