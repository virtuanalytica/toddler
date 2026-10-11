# Rekenen met eenheden bij weinig voorbeelden

**Bronnen:** de actuele [SI Brochure van BIPM](https://www.bipm.org/en/publications/si-brochure) definieert het eenhedenstelsel; de [NIST Guide to the SI](https://www.nist.gov/pml/special-publication-811) geeft praktische notatie en conversieregels. Let op dat de oudere NIST SP 811 niet alle SI-wijzigingen uit 2019 verwerkt; gebruik BIPM bij een conflict.

**Kernkennis:** een berekening heeft een getal, een grootheid en een eenheid. Optellen vereist compatibele dimensies; vermenigvuldigen en delen combineren dimensies. Een numeriek aannemelijk antwoord met een foutieve schaal of eenheid is onjuist. Controleer waar de nulpunten van temperatuurschalen meespelen: een Celsiuswaarde kan niet als gewone schaalfactor naar kelvin worden vermenigvuldigd.

**Praktijk:** bouw een kleine calculator voor lengte, tijd, massa, snelheid, energie en vermogen. Gebruik maximaal vijf zichtbare voorbeelden om een Toddler een parse- of toolaanroep te leren; leg voorbeeldselectie vast. De rekenkern moet getallen en eenheden gescheiden verwerken, conversies expliciet maken en ongeldige optellingen weigeren. Neem minstens op: `5 km / 20 min` naar `m/s`, `250 W × 2 h` naar `kWh`, en een expres foutieve `3 m + 4 s`. Vergelijk met een kale taalmodelaanpak zonder calculator onder dezelfde prompts.

**Verwachte uitvoer:** code, versie, vijf oefenvoorbeelden, exacte verwachte waarden, uitvoertrace en foutenrapport. De drie zichtbare controles geven respectievelijk `4,166666… m/s`, `0,5 kWh` en een foutmelding voor onverenigbare dimensies. Meet exact antwoordpercentage én eenheidsjuistheid afzonderlijk; leg afronding en significante cijfers vast.

**Foutencheck en toets:** de reviewer genereert nieuwe getallen, gemengde voorvoegsels en eenheden die niet optelbaar zijn. Er mag geen enkel voorbeeld of antwoord uit deze afgeschermde set in promptoptimalisatie of training terechtkomen. Een antwoord krijgt alleen krediet als waarde, dimensie en conversiestappen kloppen.
