# GUI-gronding op een onafhankelijke taak

**Bronnen:** [OSWorld, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5d413e48f84dc61244b6be550f1cd8f5-Abstract-Datasets_and_Benchmarks_Track.html) onderzoekt interactieve computeracties met uitvoeringsgebaseerde evaluatie. De [officiële OSWorld-code](https://github.com/xlang-ai/OSWorld) maakt de omgeving en publieke taken inzichtelijk. Publieke OSWorld-items zijn geen private Toddler-promotietoets.

**Kernkennis:** GUI-gronding koppelt een doel aan een zichtbaar element en een uitvoerbare actie. Het systeem moet tussen observatie en klik rekening houden met schaal, scroll, vensterpositie, overlay en toestandwijzigingen. Een correcte coördinaat op een oude screenshot kan in de nieuwe toestand gevaarlijk zijn. Beoordeel taakresultaat én de afzonderlijke acties.

**Praktijk:** bouw in een geïsoleerde browser of desktopcontainer een lokale testapp met formulieren, een dialoog en een veranderende lay-out. De agent krijgt alleen screenshots en toegestane controlleracties. Log screenshot-hash, resolutie, doeltekst, voorgestelde coördinaat, uitgevoerde actie, nieuwe toestand en pauze/redengeving voor risicovolle stappen. Maak twee zichtbare oefentaken en laat een tweede persoon minstens twaalf nieuwe varianten maken met andere schaling, scrollpositie en knoppen. De agent mag die holdout niet lezen tijdens ontwikkeling.

**Verwachte uitvoer:** een reproduceerbare omgeving, trajecten, succespercentage, miss-click-rate, aantal onomkeerbare fouten en p95-actietijd. Gebruik alleen eigen of geautoriseerde omgevingen; botdetectie ontwijken is geen kwaliteitscriterium.

**Foutencheck en toets:** wissel een knop na observatie van positie en voeg een bevestigingsdialoog toe. Een geldige agent herobserveert en stopt bij onzekerheid. De onafhankelijke evaluator bewaart taakdefinities en eindtoestandschecks apart; alleen geaggregeerde uitslagen gaan terug naar de leerling.
