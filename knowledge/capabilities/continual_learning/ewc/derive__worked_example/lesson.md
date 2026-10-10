# EWC afleiden en tegen alternatieven toetsen

**Bron:** [Kirkpatrick e.a., Overcoming catastrophic forgetting](https://doi.org/10.1073/PNAS.1611835114).

**Theorieopdracht:** leid vanuit een lokale kwadratische benadering de penalty \(\lambda/2\sum_i F_i(\theta_i-\theta_i^*)^2\) af. Verklaar de rol en beperkingen van een diagonale Fisher-benadering. Laat in een tweeparameter-tegenvoorbeeld zien dat te grote \(\lambda\) plasticiteit blokkeert.

**Praktijk:** implementeer EWC alleen in een onderzoeksbranch op een klein tweetaaknetwerk. Vergelijk met gewone fine-tuning, replay en een bevroren nieuwe expert met gelijke parameter- en computegrenzen. **Toets:** lever een sweep van \(\lambda\), vergeten én nieuwe-taak-AUC; kies de winnaar met vooraf vastgelegde regel.
