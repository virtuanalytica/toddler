# G4: tweede openbare zoekronde met eigen beloning

Na de eerste vijf-profielenronde voegde Teacher een zesde profiel toe:
4.096 PPO-stappen vanuit de actieve G3-policy, op eigen MiniGrid-observaties
en omgevingsbeloning, zonder toegang tot het volledige oracle-grid. Vier
kinderen konden daarnaast verder leren vanaf hun gehashte kandidaatgewicht
van de vorige ronde. De volledige vijfkinderenrun duurde 277 seconden en
maakte een controleerbaar manifest in
`~/.local/share/teacher/g4-search/20261010T001936Z-886c2e/`.

| Kind | Beste profiel | Openbare ontwikkeling kandidaat / G3 | Openbare controle kandidaat / G3 | Nieuwe route |
|---|---|---:|---:|---|
| t6001 | PPO op eigen beloning | 1,02028 / 1,02306 | 1,01507 / 1,02465 | nee |
| t6002 | Vorige kandidaat + oracle | 0,97910 / 0 | 0,96799 / 0 | ja |
| t6003 | Vorige kandidaat + oracle | 0,96708 / 0 | 0,99271 / 0 | ja |
| t6004 | Vorige kandidaat + oracle | 0,96868 / 0 | 0,99792 / 0 | ja |
| t6005 | Vorige kandidaat + oracle | 0,95993 / 0 | 0,93750 / 0 | ja |

**G3 blijft officieel.** De nieuwe evaluator valideerde alle vijf
gewichtbestanden en weigerde het cohort vóór creatie van private seedsets,
omdat slechts vier kinderen op openbare kaarten verbeteren. De PPO-winnaar
voor t6001 is alleen de beste *onder de geprobeerde profielen*, niet beter
dan zijn ouder.

Beide openbare kaartsets waren op dezelfde dag al voor de eerste ronde
gebruikt. Deze tweede ronde is daarom uitsluitend ontwikkelonderzoek; haar
controlekaarten zijn geen nieuwe onafhankelijke replicatie. De volgende
nacht gebruikt een verschoven openbare set. Voor een echte volgende
generatie moet het curriculum ook moeilijkere, nog niet verzadigde taken
bevatten en moet de daarvoor passende private toets vooraf worden bevroren.
