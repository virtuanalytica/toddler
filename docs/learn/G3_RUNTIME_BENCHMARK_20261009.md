# G3-recombined: runtime review, 9 October 2026

**Later status:** G3-recombined [survived a separate signed lineage review](G3_PROMOTION_20261009.md).
The decision below describes the state when this runtime probe was recorded.

**Decision:** G2 remains the official generation. The G3-recombined candidate
passed two pre-registered secret-seed comparisons and is eligible for lineage
review; the review packet is still unsigned. This change makes its frozen
task-expert router loadable and measures its CPU serving cost. It does not
train or promote a new generation.

The [trial and replication](G3_RERUN_RESULTS.md) used the same five child
weights and route on independent secret seed sets. The
[review packet](G3_RECOMBINED_REVIEW.json) contains the protocol and report
hashes and empty approval fields. `Registry.load` now verifies each saved
router's SHA-256 before loading it. The runtime probe also compared every
embedded expert tensor against the corresponding source model and checked
all 25 source weight hashes against the frozen protocol. All comparisons
passed; protocol SHA-256 is
`fbdb806fd77ee14e6519ab3efb774a484417a07649a38f3f95c59059074e359d`.

## CPU serving probe

Five matched G2 parents and five G3-recombined children were measured with
one Torch CPU thread. Each task used 128 real MiniGrid observations from a
seeded public environment walk. Each child and task had three alternating
timing rounds of 300 single-state and 40 batch-128 forwards. The values below
are medians across the five children; they are operational timings, not
secret-task scores or a statistical superiority test.

| Task | G2 single-state median | Routed candidate | G2 batch states/s | Routed candidate |
|---|---:|---:|---:|---:|
| DoorKey-8 | 0.0560 ms | 0.0580 ms | 1,303,422 | 1,262,574 |
| UnlockPickup | 0.0567 ms | 0.0579 ms | 1,280,368 | 1,263,267 |

Average saved weights were **319,662 bytes** for G2 and **1,608,117 bytes**
for the routed candidate, about **5.03×** as much. Forward latency remains
close because only one expert runs for a task, but the small observed
differences are sensitive to host load. CPU energy could not be attributed
to these forwards on a shared machine and is reported as unmeasured. The
full machine-readable result is
[`g3_recombined_efficiency_verified_20261009.json`](g3_recombined_efficiency_verified_20261009.json).
As a runtime smoke test, `G3-recombined/t5001` completed public DoorKey-8
seed 10000 in 16 steps (normalised return 1.0861). This public example is
excluded from the secret promotion comparisons.

## Next gate

UnlockPickup was still unsolved on both secret sets. A new candidate needs a
fresh registered protocol, independent training and evaluation seeds, a
matched control, old-task retention, and the same five-child promotion tests.
The stored G3-recombined candidate can enter the official lineage only after
the separately recorded human review/import decision. No code path in this
change imports its trial weights into the official registry.

Reproduce this cost probe on a machine with the registries and frozen
protocol locally available:

```bash
PYTHONPATH=. python3 scripts/benchmark_routed_successor.py \
  --baseline-root /path/to/official-registry \
  --candidate-root /path/to/trial-registry \
  --source-protocol docs/learn/PREREG_rerun_g3.json \
  --out /path/to/new-public-efficiency-report.json
```
