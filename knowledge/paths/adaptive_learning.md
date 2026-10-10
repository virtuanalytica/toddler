# Leerlijn: een adaptieve Toddler bouwen en toetsen

**Ingang:** Python, lineaire algebra, kansrekening, gradient descent en basiskennis van Gymnasium/MiniGrid. **Eindproduct:** een reproduceerbare kandidaat die op nieuwe taakvolgordes sneller leert, geërfde taken behoudt en binnen een expliciet resourcebudget blijft. Dit is een opleidings- en onderzoeksroute; G5 begint pas na een overlevende G4.

| Stap | Capability-map | Kernvraag en praktische oplevering |
|---|---|---|
| 1 | [synaptische plasticiteit](../capabilities/neuroscience/synaptic_plasticity/explain__worked_example/lesson.md) | Welke biologische observaties inspireren lokale leermodellen, en waar houdt de analogie op? Schrijf een mechanistische uitleg met beperkingen. |
| 2 | [replay](../capabilities/neuroscience/hippocampal_replay/reproduce__simulated_task/lesson.md) | Simuleer snelle opslag en uitgestelde replay; scheid biologisch bewijs van engineeringhypothese. |
| 3 | [vergeten](../capabilities/continual_learning/catastrophic_forgetting/diagnose__task_sequence/lesson.md) | Maak een taakvolgorde met revisits en meet de terugval per taak. |
| 4 | [experience replay](../capabilities/continual_learning/experience_replay/implement__simulated_task/lesson.md) | Bouw een provenance-gebonden trainingsbuffer en vergelijk met verder trainen. |
| 5 | [EWC](../capabilities/continual_learning/ewc/derive__worked_example/lesson.md) | Leid de Fisher-gewogen penalty af en toets tegen replay en een bevroren expert. |
| 6 | [modulaire experts](../capabilities/continual_learning/progressive_networks/implement__compute_budget/lesson.md) | Vergelijk een nieuwe expert met gedeelde trunk-update onder gelijke kosten. |
| 7 | [plasticiteitsdiagnostiek](../capabilities/neural_networks/plasticity_diagnostics/diagnose__task_sequence/lesson.md) | Meet dormant units en opnieuw leren na vele taakwissels. |
| 8 | [continual backpropagation](../capabilities/continual_learning/continual_backprop/implement__simulated_task/lesson.md) | Probeer selectieve unitvervanging uitsluitend op een kandidaatkopie. |
| 9 | [gepaarde toets](../capabilities/evaluation_and_causality/paired_trials/design__independent_holdout/lesson.md) | Ontwerp een verzegeld experiment dat een ouder en kandidaat eerlijk vergelijkt. |
| 10 | [deployment-gate](../capabilities/systems_and_safety/deployment_gates/validate__real_time/lesson.md) | Toon latency, rollback, lineage en foutafhandeling op de echte runtime aan. |

**Toetsing:** rapporteer leercurve-AUC op nieuwe taken, terugval op alle oude taken, benodigde interacties, geheugen/energie per verbetering en onzekerheidsintervallen. Laat een tweede persoon de run uit code en hashes reproduceren. Nul terugval op openbare kaarten of een mooie enkele demo geldt niet als promotiebewijs.
