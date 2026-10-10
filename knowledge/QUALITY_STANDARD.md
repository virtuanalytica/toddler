# Kwaliteitsgrens voor capability-materiaal

Een `capability.json` is een adres, geen bewijs. Deze vier statussen worden handmatig en controleerbaar toegekend:

1. `scaffold`: doel en locatie bestaan, maar materiaal en bewijs ontbreken.
2. `lesson_draft`: leerdoel, ten minste één primaire bron, uitleg, oefening, verwachte output en foutencheck bestaan. Dit is nog geen aangetoonde beheersing.
3. `reviewed_lesson`: een vakinhoudelijke reviewer heeft bronnen, uitleg, reproduceerbaarheid en taakveiligheid gecontroleerd; de review heeft een datum en commit-hash.
4. `assessed_capability`: een specifieke **student of agent** heeft een onafhankelijke toets gehaald. De map zelf blijft lesmateriaal; het bewijs hoort bij die student/agent, versie, model, prompts, seedband, tijd en beoordelaar in een apart auditrecord. Zet nooit een algemene `mastery_claim: true` op een catalogusmap.

Een PhD-/expert-practitioner-leerpad vraagt per module:

- **Theorie:** definities, aannames, een afleiding of formele tegenwerping, en ten minste één primaire bron plus een onafhankelijke alternatieve bron waar claims betwist zijn.
- **Praktijk:** code of experiment met exacte versie, budget, dataherkomst en een voorspelbare herstelroute bij fouten.
- **Kritiek:** negatieve controles en minstens één ablatietest; benoem waar de bron niet op Toddler overdraagbaar is.
- **Toets:** een nieuwe vraag of taak die niet bij de lesontwikkeling is gebruikt, met vooraf vastgelegde rubric. Training, oefening en geheime confirmatie mogen geen item, seedband of antwoordlekkage delen.
- **Beheer:** latency, geheugen, energie, veiligheid en rollback van de gekozen oplossing. Een betere leerscore zonder deze kosten is geen volledige expert-practitioner-uitkomst.

Voor software en agenten gebruikt de Toddler/Teacher-benchmark een aparte afgeschermde toetsset. Een les in deze atlas mag niet worden gebruikt om private promotiesets te selecteren, inspecteren of trainen. Labels als `reviewed_lesson` worden niet automatisch door een LLM of generator uitgegeven. Een model dat een les kan navertellen heeft de praktijktoets nog niet gehaald.
