# G3-recombined: ondertekende promotie, 9 oktober 2026

**Besluit:** G3-recombined is de officiële overlevende generatie. Vijf
bevroren kinderen zijn in de officiële registry opgenomen. De lineage
verifieert en bevat één selectie-event voor deze generatie: rij **41**, verdict
`survived`, rijhash
`d2496dea439a9371b8bf290308e9021ea1cd01998449b5dce1f2b917db64073d`.
De eerdere G3-sp, G3-trunk en G3-scratch blijven `extinct`.

Het vooraf vastgelegde [reviewpakket](G3_RECOMBINED_REVIEW.json) met SHA-256
`99b98ac8238654fc9cb3621eee1592602d1fc91575b5c6e479ce5b9dd2a62755`
is niet gewijzigd. De aparte goedkeuring door reviewer `knight2` is op
2026-10-09 10:14:05 UTC met een Ed25519 SSH-handtekening in de namespace
`toddler-lineage` geverifieerd. De SHA-256 van `approval.json` is
`39a0c6258c4c16e026c1382a35bbb494df9017129bfb68116e94f3369aae0a5b`;
die van de handtekening is
`9e79a8d38f4c35c659490282639697f29c642cb6f3376f76938b0a85de2fce2d`.
De goedkeuring, handtekening en vertrouwde publieke sleutel zijn met beperkte
bestandsrechten bewaard onder
`/media/knight2/EDS2/toddler-generations/G3-recombined/signed-review/`.
De privésleutel staat daar niet. De oorspronkelijke review-JSON houdt bewust
`pending_human_review`: de ondertekende beslissing is een afzonderlijk,
hashgebonden bewijsstuk.

Dit was de laatste menselijke promotiecontrole van de originator Deve Luse.
De cryptografische SSH-identiteit op het getekende bewijs is `knight2`; het
bewijsstuk zelf noemt niet de burgerlijke naam. Voor latere generaties is
een menselijke handtekening geen standaardpoort: onafhankelijke audittrials,
voorouderbehoud en een verifieerbaar lineage-event nemen die rol over.

Voor de import zijn beide protocol- en rapporthashes, de onafhankelijke
seedset-ID's, alle geregistreerde promotievoorwaarden, de vijf kandidaatgewichten
in beide proefregistries en alle 25 brongewichten opnieuw gecontroleerd.
Na de import zijn de vijf gewichtbestanden tegen hun metadata en lineage-births
geverifieerd. De lineage-hashketen geeft `(True, None)` terug.

| Afgeschermde toets | G3-recombined | G2 op dezelfde seeds | p tegen G2 | p tegen willekeurige route |
|---|---:|---:|---:|---:|
| Eerste set | 0,9458 | 0,8115 | 0,00397 | 0,01587 |
| Onafhankelijke herhaling | 0,9468 | 0,8120 | 0,00397 | 0,02778 |

De [runtimeproef](G3_RUNTIME_BENCHMARK_20261009.md) mat bij DoorKey-8 met
één CPU-thread 0,0580 ms per toestand, tegenover 0,0560 ms voor G2. De
opgeslagen gewichten zijn ongeveer 5,03 keer zo groot. **UnlockPickup bleef
onopgelost** op beide geheime sets. CPU-energie is niet gemeten. Er zijn geen
nieuwe trainingsstappen uitgevoerd: de generatie combineert bestaande,
bevroren experts per taak. De geheime seeds en salt zijn niet gepubliceerd of
voor routekeuze gebruikt.

Het [ontwikkelverslag](reports/G3-recombined_ontwikkelverslag.pdf) is pas na
het `survived`-verdict gegenereerd en bevat de lineage, hashcontrole en de
afgeschermde geaggregeerde resultaten. Het dashboard toont de gepaarde G2-rijen
voor de afgeschermde sets; historische openbare G2-punten zijn geen directe
controle voor G3's geheime punten.
