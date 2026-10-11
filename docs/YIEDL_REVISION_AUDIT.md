# Dagelijkse YIEDL-revisiecontrole

Een datum in de YIEDL-historie bewijst niet dat de waarde op die dag al beschikbaar was. Deze audit legt vanaf **11 oktober 2026** elke dag een onveranderlijke momentopname vast van de laatste 90 kalenderdagen: `date`, `symbol`, PVM-, onchain- en sentimentgemiddelde. De volgende opname vergelijkt dezelfde token/datum-sleutels en meldt gewijzigde waarden, later toegevoegde sleutels en verdwenen sleutels.

De eerste opname bevat 204.342 rijen en heeft alleen status `baseline_capture`. Pas na volgende opnamen kan de code revisies aantonen of uitsluiten in het overlappende venster. Ook een reeks zonder revisies kan historische backfills van vóór 11 oktober niet certificeren. Voor echte point-in-time toelating moet bovendien het publicatiemoment van de oorspronkelijke YIEDL-release tegenover de Numerai-deadline worden bevestigd.

```bash
python3 -m scripts.audit_yiedl_revisions
python3 -m pytest -q tests/test_yiedl_revision_audit.py
```

De standaardbron blijft in de bestaande Numerai-repository; opnamen en hashrapporten staan op claude-data onder `finance-agent-runs/yiedl_source_audit/`. Er gaat geen YIEDL-brondata naar Git. De audit heeft een exclusieve lock, controleert unieke sleutels en weigert een bron die meer dan zeven dagen achterloopt. Een herhaling met dezelfde bron-hash en datum is idempotent.

Voor installatie na merge op de vaste Toddler-checkout, na de bestaande YIEDL-ingest van 12:15 lokale tijd en na de gebruikelijke publicatietijd:

```cron
30 15 * * * cd /media/knight2/claude-data/knight1/toddler-knowledge-production && /usr/bin/python3 -m scripts.audit_yiedl_revisions >> /media/knight2/claude-data/knight1/finance-agent-runs/yiedl-source-audit-cron.log 2>&1
```

Een status `revision_detected` zet de YIEDL-kandidaat terug naar onderzoek; de nulmeting blijft als bewijs bewaard. Beoordeel de wijziging per bronveld en datum voordat een model opnieuw wordt getraind.
