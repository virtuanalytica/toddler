# Experience replay met aantoonbare herkomst

**Bron:** [Rolnick e.a., CLEAR](https://proceedings.neurips.cc/paper_files/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html). CLEAR combineert oude ervaringen met nieuw gedrag en beperkt vergeten in de onderzochte RL-omgeving; het effect is nog niet op Toddler aangetoond.

**Theorieopdracht:** verklaar hoe de verdeling van oude en nieuwe trajecten de gradient beïnvloedt. Beschrijf risico's van niet-stationaire policy-data en waarom een loutere replaybuffer geen eerlijke private toets is.

**Praktijk:** implementeer op een kandidaatkopie een begrensde buffer met taak, seedband, modelhash, actie, beloning en tijd. Sluit alle private/promotieseeds expliciet uit. Vergelijk onder identiek trainingsbudget `new_only`, `replay_only` en `mixed`. **Toets:** lever de bufferhash, runscripts, oude-taak-retentie, nieuwe-taak-AUC en een ablatietabel; herstel de ouder zonder geheugenmigratie.
