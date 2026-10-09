# Technical audit: Toddler brain, JEV, CLM and the GPU model motor

Status: 9 October 2026. This is a code and measurement audit, not a claim that
all components run as one production system. Runtime checks are a snapshot and
may change. Source of truth for future promotions is a new private, matched
evaluation, not the archived public composite.

## Two distinct learning systems

Toddler's official **G3-recombined** is a frozen task-expert router for Gym and
MiniGrid policies. `TaskExpertRouter` maps every registered task to one
complete network. It passed two private seed gates and is in the lineage
registry. It is not a language model, a GPU mixture, or a learning router that
changes weights while answering a chat.

The proposed **language-model motor** is separate: a chat/proxy can ask several
local models for drafts, then an aggregator produces the final answer and tool
calls. Toddler's MCP server exposes policy tools to that chat. A live model
answer can explain a policy episode, but this does not update G3 weights or
prove that Toddler + Teacher + an agent improves software work.

```text
input / task
   │
   ├─ deterministic STOP rules ────────────────────────────────┐
   │                                                           │ veto
   ├─ future physical state → hard sensor limits → JEV answers ┘
   │                                 └→ CLM alternative: not wired
   │
   ├─ policy task → frozen G3 task-expert route → environment action
   │
   └─ language task → model router / MoM proxy
                      ├→ Ada proposer(s): drafts, in parallel
                      └→ V100 aggregator: final text / tool calls
                           → tool result → verification / audit
```

The physical reflex and language-model path are **not** one end-to-end control
loop today. The static `toddler.specialize.route` ranks experts by a supplied
quality-minus-cost score; it is not the live MoM proxy and has no promoted
private router score.

## Implemented reflex and its limits

| Component | Implemented | Important limit |
|---|---|---|
| Hard STOP | `toddler/stop.py` vetoes secrets and specified external/contact actions before utility is compared. | The caller must classify the candidate action correctly. |
| Sensor threshold | `toddler/fastpath.py` checks force, speed and person distance first. | These are demonstration defaults, not a robot certification. Non-finite sensor values are not explicitly rejected by the present comparisons. |
| Typed JEV questions | Four defaults: grasp slip, person in zone, imminent collision and unstable object. `toddler/jev.py` sends one TypeSafe-style request and parses all requested answers. | The extra four risks mentioned in earlier web prose are ideas, not default code. |
| Decision | A hard limit, missing answer, invalid probability or reply after 20 ms yields STOP; otherwise the code compares `probability + uncertainty` with slow/stop thresholds. | A non-finite elapsed time is not explicitly rejected. The `noul` uncertainty default (0.05) is a configured margin, not a measured safety interval. |
| JEV service | `jevserver` exposes the typed API and uses a local llama.cpp probability backend. | Health of the API is not proof that the model backend works or meets 20 ms. |
| CLM | Upstream CLM provides typed decisions and candidate ranking through a compatible API. | No CLM adapter, server or local reflex calibration was found in this Toddler checkout. |

The JEV documentation reports **1,385 ms cold** and **5 ms cache-hit** latency
for four questions. It also reports PIQA calibration ECE improving from 0.252
to 0.066 on 150 validation items (300 yes/no questions). Neither figure
establishes calibration for physical hazards, and the cold path misses the
20 ms budget. At this audit's runtime check, the JEV API health endpoint was
up while its configured llama.cpp backend was absent. A cache hit or health
check must therefore never stand in for an end-to-end reflex test.

Upstream [Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM) documents
a Qwen3-8B pooling encoder plus a small contrastive head. It caches action
embeddings and exposes typed decisions. Its published speed and verifier
results are upstream results on other hardware/tasks. Local latency, VRAM,
hazard recall and calibration on Toddler states remain unmeasured. CLM is a
plausible **candidate ranker or verifier**, not an independently certified
physical safety authority.

## GPU motor: topology and evidence

This host has **two V100-SXM2 32 GB** cards (logical GPU 3 and 4, NVLink pair)
and **four RTX 4000 Ada 20 GB** cards (0, 1, 2 and 5). The checked
`virtualv_llm/infra/model_serve_configs/mom-live.sh` configuration assigns
the V100s two Qwen3.8-27B Q4 aggregator replicas; Ada 0 hosts Devstral 24B,
Ada 1 and 5 host Qwen3.5-27B replicas, and Ada 2 hosts Gemma4-26B-A4B.
`mixture_proxy.py` collects drafts in parallel and sends them to the
aggregator. Tool-result turns go directly to the aggregator. Replicas spread
requests; they do not multiply a single answer's token rate.

| Candidate | Measured or documented fact | Conclusion |
|---|---|---|
| Existing `mom-live-4` with Qwen3.8-27B Q4 replicas | Historical 2026-10-05 run: public composite 0.885, specialist 0.720, 12-item holdout 0.750. | Functioning MoM reference, but the public composite is archival and the holdout is too small to promote a new Toddler motor. |
| Qwen3.8 Flash-Next AP-IQ2_S GGUF | `virtualv_llm` roadmap reports about 40.8 generated tokens/s on the V100 pair and 40.6 on four Ada cards in separate tests. | Measured fast single-model route; a Flash-Next aggregator with the Ada proposers still needs a matched mixture run. It is **not** 1Cat TP2. |
| Qwen3.8-27B QUASAR NVFP4, 1Cat TP2 | A separate 1Cat TP2 MoM run on the same V100 pair scored 0.578 on the archived public composite versus 0.885 for the Q4-replica mixture. Later 1.5.1 stability checks improved, but the full TP2 status marks a failed quality canary and promotion ineligible. | A real TP2 experiment, but a different Qwen model/weight format from Flash-Next and not a quality-approved core. |
| Qwen3.8 Flash-Next NVFP4, 1Cat TP2 | Local 1.5.0 attempt in the roadmap ran out of V100 memory. The upstream 1.5.1 release gives a **TP4 command for four peer-connected V100 32 GB cards**; this host has two. | No verified local Flash-Next TP2 throughput or end-to-end quality claim. A mixed V100/Ada TP2 scheme would require separate kernel, memory and quality validation. |

The 2026-10-05 like-for-like *human-eval + specialist + audit* phase reported
2.16 GPU-board Wh/answer for the Q4-replica mixture versus 2.90 for the TP2
mixture, with 4.19 versus 3.51 answers/minute. These are historical
**GPU-board** measurements, not total wall-power, and apply only to that
matched phase. Never compare a whole suite with a shorter phase or confuse
model tokens/s with complete mixture answers/minute.

At the 9 October audit snapshot, the MoM and Qwen TP2 services were inactive;
GPU occupancy was near zero except another workload. No continuously running
six-GPU Toddler motor was observed. The JEV service was running, but its model
backend was not. This status can change without a repository edit.

## Engineering decision

1. Keep the measured Q4-replica mixture as the **historical reference**. A
   Qwen3.8 Flash-Next GGUF aggregator on the V100 pair plus Ada proposers is
   the clearest six-GPU *candidate*. Do not label it superior until it passes
   the same private specialist, contamination-resistant, 256-token,
   latency and GPU-board-energy protocol as its comparator.
2. Treat Qwen3.8-27B 1Cat TP2 and Flash-Next TP2 as different experiments.
   Require a valid end-to-end quality canary before either is a motor. The
   upstream four-V100 Flash-Next configuration cannot be copied unchanged to
   this two-V100 host.
3. Make the physical reflex truly fail closed for NaN/infinite sensor values
   and elapsed time; independently measure hazard recall/calibration and cold
   p95/p99 latency. A cached 5 ms answer is not enough.
4. Add CLM only as a separate typed-decision candidate behind the same
   interface, with a dedicated encoder placement and local safety/latency
   tests. Compare JEV, CLM, deterministic limits and their disagreements;
   deterministic STOP remains authoritative.
5. Record the exact model ID, quant, engine version, GPU assignment,
   protocol ID and per-answer energy with every test. Keep public scores
   historical; official promotion uses fresh independently held items and
   an auditable ancestor gate.

## Evidence

- Toddler: [STOP](../../toddler/stop.py),
  [reflex](../../toddler/fastpath.py), [JEV client](../../toddler/jev.py),
  [JEV measurements](../jev/README.md), [G3 router](../../toddler/learn/routing.py),
  [specialisation router](../../toddler/specialize.py).
- VirtualV: [live MoM configuration](https://github.com/virtuanalytica/virtualv_llm/blob/main/infra/model_serve_configs/mom-live.sh),
  [proxy](https://github.com/virtuanalytica/virtualv_llm/blob/main/scripts/benchmarks/mixture_proxy.py),
  [mixture lessons](https://github.com/virtuanalytica/virtualv_llm/blob/main/docs/LESSONS_LIVE_MIXTURE_20261005.md),
  [model roadmap](https://github.com/virtuanalytica/virtualv_llm/blob/main/docs/MODEL_TEST_ROADMAP.md),
  [TP2 status](https://github.com/virtuanalytica/virtualv_llm/blob/main/reports/qwen38_tp2_full_status_20261009.json).
- Upstream: [CLM reference implementation](https://github.com/Contrastive-LM/CLM);
  [1Cat-vLLM 1.5.1 release and TP4 Flash-Next recipe](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.1).
