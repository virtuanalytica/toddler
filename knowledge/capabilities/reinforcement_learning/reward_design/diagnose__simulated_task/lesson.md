# Beloningsontwerp: diagnose van proxy-optimalisatie

**Bronnen:** [Roy e.a., ICML 2022](https://proceedings.mlr.press/v162/roy22a.html) onderscheiden expliciete gedragsbeperkingen van alleen een beloningsfunctie. [Reddy e.a., ICML 2020](https://proceedings.mlr.press/v119/reddy20a.html) onderzoeken hoe een beloningsmodel op hypothetisch gedrag kan worden bevraagd om onveilige toestanden en reward hacking te vinden. Dit zijn onderzoeksresultaten, geen garantie dat één constraint-methode iedere fout oplost.

**Kernkennis:** een proxyreward is een meetbare vervanger voor de werkelijke bedoeling. Optimalisatie kan een hoge proxywaarde vinden terwijl de bedoelde taak faalt. Maak daarom de bedoelde uitkomst en verboden gedragingen onafhankelijk van de trainingsreward observeerbaar. Een constrained MDP maakt een deel van die grenzen expliciet; de gekozen constraints kunnen zelf onvolledig zijn.

**Praktijk:** bouw een kleine, gezaaide gridworld met een doeltegel, een verboden tegel en een herhaalbare actie die proxyreward oplevert zonder het doel te bereiken. Train of programmeer twee policies bij gelijk aantal stappen: (A) optimaliseer alleen de proxy, (B) optimaliseer met een expliciete overtredingsgrens. Bewaar omgevingversie, seeds, rewarddefinitie, action traces en policies. Meet per seed: doelbereik, proxyreturn, overtredingen en echte taakscore. Presenteer minstens één trace waarin proxyreturn en taakscore uiteenlopen.

**Verwachte uitvoer:** een uitvoerbaar experiment, een tabel per seed en een korte diagnose van de exploit. Een geldige reparatie verbetert doelbereik zonder verborgen overtredingen; een mooie gemiddelde reward is onvoldoende.

**Foutencheck en toets:** een onafhankelijke reviewer wijzigt één omgevingseigenschap of rewardgewicht buiten de zichtbare oefenseeds. Voorspel vooraf welke policy zal falen, laat beide lopen en verklaar een tegenvoorbeeld. De nieuwe seedband en antwoordcriteria blijven buiten de publieke les. Test uitsluitend in de simulatie; leid hieruit geen veiligheid voor fysieke robots af.
