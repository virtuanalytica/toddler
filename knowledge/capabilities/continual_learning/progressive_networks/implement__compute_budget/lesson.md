# Nieuwe experts zonder geërfde routes te overschrijven

**Bron:** [Rusu e.a., Progressive Neural Networks](https://arxiv.org/abs/1606.04671). Het principe van bevroren oude kolommen is relevant; de huidige Toddler-route gebruikt complete taakexperts en nog geen geleerde laterale verbindingen.

**Theorieopdracht:** vergelijk parameter- en inferentiegroei voor één gedeelde trunk, één nieuwe adapter en één compleet nieuw expertmodel na (n) taken. Verklaar wanneer modulaire bescherming ten koste van transfer gaat.

**Praktijk:** maak een kandidaat voor een nieuwe taak met (a) trunk-update, (b) bevroren ouder + adapter, (c) bevroren ouder + complete expert. Meet kwaliteit, oude-taak-retentie, CPU-latency, RAM en GPU-board Wh volgens dezelfde definitie. **Toets:** verantwoord de gekozen architectuur onder een vooraf opgegeven budget; behoud de oude route als fallback.
