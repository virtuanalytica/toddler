# Her-G3: task-expert recombination, 8 October 2026

**Later status (9 October 2026):** the [signed review and import](G3_PROMOTION_20261009.md)
made G3-recombined the official surviving generation. The trial results below
remain the frozen, pre-promotion evidence.

**Verdict:** a five-child composite successor passed the promotion gate on **two independent
secret seed sets**, without changing its route or source weights between tests. It is
**eligible for lineage review**. The first trial is held in
`/media/knight2/EDS2/toddler-g3-recombined-20261008/registry`; its unchanged-weight
replication is in `/media/knight2/EDS2/toddler-g3-recombined-replication-20261008/registry`.
Both are separate from the official generation registry. G2 remains the official line until
a signed review/import decision.

The candidate recombines already trained G2, G2-scratch and three G3 policies. No new
environment training was performed. For each child, all four original G1 tasks stay routed
to that child's exact G2 policy. On the five newer tasks a frozen algorithm selected another
expert only if its **public** task mean exceeded G2 by at least 0.05. A seeded random route
through the same expert bank was frozen as the control. The route, 25 source weight hashes,
comparison rule and secret seed commitment were written to `PREREG_rerun_g3.json` **before**
the first secret evaluation. The replication protocol kept routes, source weights and gate
identical and committed a fresh seed set before its evaluation.

| Fresh secret comparison, five children per arm | Candidate | Reference | One-sided Mann–Whitney p | P(improvement) | Pass? |
| --- | ---: | ---: | ---: | ---: | --- |
| First set: candidate vs G2 | 0.9458 | 0.8115 | 0.00397 | 1.00 | yes |
| First set: candidate vs random route | 0.9458 | 0.8025 | 0.01587 | 0.92 | yes |
| Replication: candidate vs G2 | 0.9468 | 0.8120 | 0.00397 | 1.00 | yes |
| Replication: candidate vs random route | 0.9468 | 0.7990 | 0.02778 | 0.88 | yes |

The test used the existing `decide_promotion` gate: at least five runs, p < 0.05 and
P(improvement) >= 0.75. The descriptive aggregate IQM over all child-task cells was 0.9449
for the candidate, 0.7719 for G2 and 0.8433 for the random route; on replication these were
0.9465, 0.7725 and 0.8400. It also exceeded each
single source generation on this aggregate metric (best source: G2-scratch, 0.8827).

The gain is concentrated in DoorKey-8, KeyCorridor and LavaCrossing. **UnlockPickup remained
at 0** on the secret battery for all compared arms, even though one public score had favored
a G3 expert there. This is a visible public-to-secret transfer failure and a priority for the
next curriculum; it was not used to change the frozen route after evaluation. The candidate
also uses more stored weights than one shared network: the five candidate files averaged
1.53 MiB each, versus 0.30 MiB for their G2 parents. A sampled public-seed DoorKey-8 episode
loaded from the saved trial registry finished in 18 steps with normalised score 1.083.
Serving latency and energy still need a separate benchmark before deployment.

Protocol SHA-256: `fbdb806fd77ee14e6519ab3efb774a484417a07649a38f3f95c59059074e359d`.
Replication protocol SHA-256: `b7b7d8277da2236fe1516f7e6f47a4cc1964267d4777c1ebea543334342bc8ca`.
The two secret commitments are `e756a6526f000ef7e207e3e974c3388991770e35f53da32c2443db3cc7dda7a8`
(set `20261008T175942Z-5b684c`) and
`339ea961400007b6c9f22aa28a4f5b423e40c674bb6a943f963fe4b36280ed02`
(set `20261008T180515Z-b53769`), 30 seeds each, reveal after 2026-10-15. Seeds and salt
remain outside Git. The raw reports are copied without secret seeds to
`generation_g3_recombined_trial.json` and `generation_g3_recombined_replication.json`.
Both trial registries' ten weight files were loaded again and their recorded SHA-256 hashes
verified. The candidate weight hashes were identical across trials; the
official lineage ledger still verifies and still records G2 as survived.
The unsigned review packet `G3_RECOMBINED_REVIEW.json` lists the two protocol/report hashes,
five candidate weight hashes, outcomes and known limitations. Its approval fields remain empty.

Reproduce the code path with `PYTHONPATH=. python3 scripts/rerun_g3_recombination.py run
docs/learn/PREREG_rerun_g3.json --out NEW_TRIAL_DIRECTORY`. A fresh trial needs a new
protocol and seed set; the completed secret set must not be reused for route selection.
