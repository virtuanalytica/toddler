# Vergeten meten op een taakvolgorde

**Bronnen:** [CLEAR, NeurIPS 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html) en [Continual World, NeurIPS 2021](https://papers.nips.cc/paper_files/paper/2021/hash/ef8446f35513a8d6aa2308357a268a7e-Abstract.html).

**Theorieopdracht:** definieer prestatie na taak (j) op eerder geleerde taak (i), en bereken maximale terugval per taak. Scheid vergeten van plasticiteitsverlies: een netwerk kan oude taken behouden maar niets nieuws meer leren, of snel leren en tegelijk veel vergeten.

**Praktijk:** train A→B→C→A op publieke MiniGrid-varianten met vaste interactiebudgetten. Meet alle taken na iedere fase, ook wanneer ze niet meer getraind worden. Leg trainings- en evaluatieseeds apart vast. **Toets:** diagnoseer een onbekende leercurvematrix en kies welke extra meting de twee failure modes uit elkaar haalt.
