# Herkomst van agentgeheugen met echte gegevens

**Bronnen:** de W3C-aanbevelingen [PROV-O](https://www.w3.org/TR/prov-o/) en [PROV-DM](https://www.w3.org/TR/prov-dm/) beschrijven onder meer entiteiten, activiteiten, verantwoordelijke actoren en afleidingsrelaties. Gebruik ze als gegevensmodel; ze bewijzen op zichzelf niet dat een opgeslagen claim waar is.

**Kernkennis:** een geheugenfeit is een versiegebonden bewering over een bron, niet een losse zin. Leg voor iedere extractie vast welke bronversie werd gebruikt, door welke activiteit een afgeleide entiteit ontstond en wie of wat verantwoordelijk was. Een correctie van de bron moet alle afhankelijke herinneringen vindbaar maken. Herkomst en waarheidscontrole zijn aparte vragen.

**Praktijk:** kies tien publiek deelbare documenten of een toegestane interne dataset. Noteer licentie, toegang en retentie vóór ingestie. Maak per document een `Entity` met contenthash; per extractierun een `Activity` met codeversie en tijd; per feit een afgeleide `Entity` met `wasDerivedFrom`, `wasGeneratedBy` en een bronspan. Bouw een query die voor een antwoord de bronketen toont. Corrigeer daarna bewust één document en geef alle mogelijk verouderde feiten en afgeleide antwoorden terug. Voeg een verwijderverzoek toe dat zowel primaire als afgeleide kopieën in de eigen testopslag aanwijst.

**Verwachte uitvoer:** een klein PROV-compatibel grafmanifest, een herkomstquery, een correctielog en een controle op verweesde feiten. Meet percentage feiten met geldige bronhash en percentage afgeleide records gevonden na correctie. Bewaar gevoelige documenten niet in Git.

**Foutencheck en toets:** laat een reviewer een extra document toevoegen met twee tegenstrijdige versies. De leerling moet het conflict zichtbaar laten, de geldige tijd aangeven en een niet-herleidbaar antwoord afwijzen. De reviewer houdt dit document buiten de openbare oefenmap. Een mooie grafvisualisatie zonder herstelbare bronketen slaagt niet.
