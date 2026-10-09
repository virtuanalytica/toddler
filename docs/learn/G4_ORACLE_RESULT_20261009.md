# G4-oracle: eerste afgeschermde cohortproef, 9 oktober 2026

**Verdict: niet gepromoveerd.** Vijf bevroren kinderen werden op twee nieuwe,
onafhankelijke 30-seedsets vergeleken met hun G3-ouders en alle overlevende
voorouders (G1, G2 en G3-recombined). De negen taken en `sample`-modus waren
identiek voor alle armen. De eerste proef miste de vooraf vastgelegde grens
`p < 0,05`; de tweede slaagde. Er waren geen regressies onder de
voorouderbehoudsgrenzen. Beide proeven moesten slagen, dus er is geen
`G4-oracle`-import of `survived`-event in de officiële registry.

| Geheime set | Kandidaat IQM | Sterkste voorouder, G3 | Kans op verbetering | p | Poort |
|---|---:|---:|---:|---:|---|
| 1 | 1,0421 | 0,9462 | 0,82 | 0,05804 | faalt |
| 2 | 1,0427 | 0,9453 | 0,86 | 0,03746 | slaagt |

De identieke G3-ouder is ook de gematchte controle; daarom zijn de p-waarden
tegen voorouder en controle hier gelijk. Kind t6001 behield vooraf zijn G3-route
voor `unlockpickup`; bij de andere vier werd de oracle-specialist gekozen met
de vastgelegde openbare ontwikkelregel. Deze keuze is na bevriezing niet
aangepast.

Protocol-SHA-256:
`85001b20d482ced98e823d4b3395af92e60ad6b56ae7e70ed7dcc26a67df620e`.
De private trial-bestanden hebben SHA-256
`f5f60f457542c6609c0a8775f935b7099fd8f442e6078e18c39a215c81f42b52`
en `5a3acfb0ecfb55bc265a82953b21cb5e023fc2fad6a74103c51bdf4923e68f40`.
Hun individuele rijen blijven buiten Git tot de onthuldatum.

De set-ID's zijn `20261009T162828Z-b2d8d2` en
`20261009T162828Z-497329`. Hun seedcommitments zijn respectievelijk
`ac82c71c894f91aee3769b09c6a58f5657b5f96e4a1fecb053c55c4a45f4ce01`
en `802cc187380449abe9dd94cd4828d492103be0653ef74e362e6c9735e982ef5e`.
De private seeds en salt worden pas na **16 oktober 2026** onthuld; geen van
beide is gebruikt om de volgende trainingsreceptuur te kiezen.

Een volgende G4-kandidaat moet vóór een nieuwe toets weer volledig worden
bevroren en twee **nieuwe** geheime seedsets krijgen. De huidige test blijft
als mislukte maar informatieve trial in het auditarchief.

Een CPU-microproef met één thread en 2.000 herhaalde toestanden mat voor kind
t6004 een mediaan van 0,0603 ms per `unlockpickup`-beslissing, gelijk aan zijn
G3-ouder; `doorkey8` bleef eveneens circa 0,060 ms. Het gewichtsbestand groeide
van 1.608.066 naar 1.929.810 bytes (+20%). Dit meet alleen policy-forwardtijd,
geen volledige spel- of systeemenergie.

Na bevriezing is op uitsluitend **openbare** ontwikkelkaarten een apart
onderzoeksvoorstel getest: het eerste kind starten vanuit de al sterke G3-sp
`unlockpickup`-expert en daarop 128 oracle-demonstraties trainen. Dat gaf
50/50 en mean 1,0240, tegenover 50/50 en 1,0148 voor zijn G3-ouder.
Dit gewicht behoort niet tot de afgeschermde cohortproef. Een eventuele nieuwe
kandidaat vereist een nieuw vooraf vastgelegd protocol met een expliciete
correctie voor opeenvolgende pogingen en volledig nieuwe geheime sets.
