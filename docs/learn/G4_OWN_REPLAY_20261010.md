# G4: eigen ervaring herhalen na een geleerde specialist

**10 oktober 2026 — openbare ontwikkelproef, geen promotiebewijs.** Toddler
heeft nu een [begrensde replaybuffer](../../toddler/learn/experience_replay.py)
voor voltooide, succesvolle eigen trainingsepisodes. Iedere rij bevat taak,
trainingsseed, brongewichthash, observaties, acties, beloningen per stap,
episode-opbrengst en verzameltijd. Het archief heeft hashes per traject en
voor het geheel; inladen weigert een gewijzigd archief. De private seedband
wordt al vóór het openen van de omgeving geweigerd. PPO gebruikt deze
off-policy data niet; uitsluitend een kandidaatkopie leert via imitatie van
haar eigen succesvolle acties. De gedeelde trunk en andere taakmodules
blijven bevroren.

De proef begon bij de vier bestaande, gehashte `unlockpickup`-specialisten
uit Teacher-cohort `20261010T001936Z-886c2e`. Per kind werden 128 nieuwe
episodes op trainingsseeds verzameld, maximaal 16.000 stappen bewaard en
vier BC-epochs gedraaid met `lr=1e-4`. Dezelfde dertig **bekende openbare**
seeds 10000–10029 werden vóór en na gemeten met sampled acties en dezelfde
random-ankerwaarde. Ruwe per-seedscores, trajectarchieven en kandidaatgewichten
staan buiten Git onder
`~/.local/share/teacher/g4-own-replay-v3-20261010/`; alle archieven en
gewichthashes zijn opnieuw geverifieerd. De replaymodule had SHA-256
`c29ceb70131bb995ca8a4f5d4b8f76f8d102f225a86e96c257f08f8d994609ec`.

| Kind | Succesvolle trainingsepisodes | Bewaarde stappen | Voor / na | Verschil |
|---|---:|---:|---:|---:|
| t6002 | 118 | 3.924 | 0,93414 / 0,90336 | −0,03079 |
| t6003 | 115 | 2.714 | 1,02928 / 1,02743 | −0,00185 |
| t6004 | 126 | 3.353 | 0,98137 / 0,99988 | +0,01852 |
| t6005 | 126 | 4.042 | 0,96400 / 0,91574 | −0,04826 |

Dit mechanisme levert in deze opzet **geen betrouwbare winst**. Een
succesvolle episode bevat ook omwegen; alle acties daaruit opnieuw nadoen
kan die omwegen versterken. Dat is een mogelijke verklaring, geen bewezen
oorzaak. De vier kinderen, één trainingsseed per kind en hergebruikte publieke
evaluatiekaarten geven geen schatting van generatiewinst. De eerdere PPO-proef
gebruikte een ander interactiebudget en mag niet als eerlijke winnaar van
deze vergelijking worden uitgeroepen.

Een volgende onderzoeksfamilie kan eerst een vooraf gekozen filter voor
actievoordeel of trajectkwaliteit testen, met gelijke interactie- en
rekenbudgetten, meerdere trainingsseeds, een verse openbare ontwikkelset
en een afzonderlijke controleset. Alleen een daarna bevroren kandidaat mag
naar nieuwe private bevestiging. De huidige G4-nachtproef en G3 blijven
ongewijzigd.
