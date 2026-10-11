# Meertalige betekenis bij domeinverschuiving

**Bronnen:** de makers documenteren [FLORES-200](https://github.com/facebookresearch/flores/blob/main/flores200/README.md) als publiek meertalig evaluatiecorpus met dev/devtest en een verborgen testdeel; het [NLLB-onderzoek](https://arxiv.org/abs/2207.04672) combineert meertalige automatische en menselijke evaluatie. Publieke FLORES-zinnen zijn oefen- of referentiemateriaal, geen afgeschermde promotietoets voor moderne modellen.

**Kernkennis:** vertaalbaarheid, taalkundige vloeiendheid en feitelijke behouding zijn verschillende eigenschappen. Een domeinverschuiving kan termen, register, namen en getallen veranderen terwijl zinsvormen bekend blijven. Een automatische metriek kan zulke fouten missen; laat daarom menselijke beoordelaars betekenis, terminologie en weglatingen apart annoteren.

**Praktijk:** vergelijk twee lokale modellen of prompts voor Nederlands ↔ Engels en, alleen met een bevoegde moedertaalreviewer, een Kantonese variant. Gebruik een publieke oefenset uit FLORES of zelfgeschreven toegestane teksten, en een apart door een reviewer gemaakte set van 30 korte zinnen uit één nieuw domein. Bevries modelversies, decodeerinstellingen en voorbeeldkeuze. Label per zin behoud van namen/getallen, betekenis, register en onterechte toevoegingen. Rapporteer per taalrichting en domein, inclusief annotatorverschil.

**Verwachte uitvoer:** datasetkaart met licentie en splits, modelmanifest, gepaarde resultaten en fouttaxonomie. Bewaar de nieuwe toetszinnen buiten de repository en stel ze niet aan de kandidaat of promptverbeteraar beschikbaar.

**Foutencheck en toets:** geef beide modellen een nieuwe domeinbatch met terminologie en numerieke negatie. Een onafhankelijke beoordelaar beslist vooraf de rubric. Claim geen taalbeheersing op basis van een gemiddelde over talen; zonder Kantonese reviewer blijft dat onderdeel ongetoetst.
