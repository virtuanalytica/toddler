# Verlies van leervermogen diagnosticeren

**Bron:** [Dohare e.a., Nature 2024](https://www.nature.com/articles/s41586-024-07711-7). De studie onderscheidt plasticiteitsverlies van catastrofaal vergeten en rapporteert dormant units en andere representatiesignalen.

**Theorieopdracht:** formuleer een plasticiteitsmaat uit een vaste nieuwe-taak-leercurve onder gelijk data- en compute-budget. Verklaar waarom nul gradient, lage effectieve rang of veel slapende units aanwijzingen zijn, maar afzonderlijk geen oorzakelijk bewijs.

**Praktijk:** meet per laag activaties, gradientnormen, effectieve rang en nieuwe-taak-AUC na opeenvolgende trainingsfasen. Vergelijk met een vers geïnitialiseerde controle van dezelfde grootte. **Toets:** verklaar een geval waarin de oude taakscore gelijk blijft terwijl de nieuwe-taak-AUC daalt; reproduceer over meerdere initialisaties.
