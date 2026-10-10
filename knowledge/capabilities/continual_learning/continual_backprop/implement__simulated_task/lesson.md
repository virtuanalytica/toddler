# Selectieve unitvervanging op een kandidaatkopie

**Bron:** [Dohare e.a., Loss of plasticity in deep continual learning](https://www.nature.com/articles/s41586-024-07711-7). Hun continual-backprop-methode herinitialiseert een klein deel weinig gebruikte eenheden; hun instellingen zijn geen Toddler-hyperparameters.

**Theorieopdracht:** beschrijf selectie op gebruik, leeftijd van een unit en de gevolgen van het resetten van inkomende én uitgaande gewichten. Vergelijk met L2 en shrink-and-perturb.

**Praktijk:** maak drie identieke kandidaattrainingen: geen reset, willekeurige reset en gebruiksgebonden reset. Bevries de officiële ouder. Sweep alleen op openbare ontwikkeling; log hoeveel units zijn vervangen en meet nieuwe-taak-AUC, terugval en variatie tussen zaden. **Toets:** een onafhankelijke set moet de gekozen variant bevestigen; bij regressie blijft de ouder actief.
