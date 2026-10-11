# Energie per antwoord reproduceerbaar meten

**Bronnen:** [NVIDIA DCGM-velddefinities](https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html) beschrijven onder andere GPU-board power in watt en een totale GPU-energiecounter in millijoule sinds driverreload. De [Linux Power Capping Framework-documentatie](https://www.kernel.org/doc/html/latest/power/powercap/powercap.html) beschrijft `energy_uj` voor ondersteunde zones. Geen van beide is automatisch een meting van het hele systeem aan het stopcontact.

**Kernkennis:** energie is een integraal van vermogen over tijd. `Wh = ΔJ / 3600`; een counter in mJ wordt eerst door 1000 gedeeld. Meet interval, idle-baseline, concurrency en counterreset. Tel hiërarchische CPU-zones niet dubbel. GPU-board Wh, CPU-zone Wh en extern gemeten systeem-Wh zijn aparte grootheden.

**Praktijk:** draai één vast promptpakket van ten minste tien identieke decodeerconfiguraties onder een vooraf gekozen tijd- en tokenbudget. Neem voor en na iedere vraag DCGM-counterwaarden op; noteer GPU-UUID, driver, modelhash, prompt/outputtokens en tijd. Meet een even lang idle-interval. Als een externe wandmeter beschikbaar is, log die apart met type en kalibratiestatus. Rapporteer bruto board-Wh, idle-gecorrigeerde board-Wh, token/s en kwaliteit per antwoord. Ontbreekt de energiecounter, gebruik dan gedocumenteerde power-samples met samplefrequentie en foutmarge; label het als schatting.

**Verwachte uitvoer:** ruwe tijdstempels en eenheden, replay-commando, berekening per vraag, mediangen en spreiding. Een negatieve counterdelta door reset of wrap maakt die meting ongeldig. Kies geen snellere preset tijdens dezelfde gepaarde vergelijking.

**Foutencheck en toets:** een reviewer injecteert één reset, één parallelle GPU-taak en een mJ/J-verwisseling in een nieuwe log. De leerling moet alle drie detecteren en de betreffende metingen uitsluiten of herstellen. Een GPU-boardgetal mag nooit als totale systeemkosten worden gepresenteerd.
