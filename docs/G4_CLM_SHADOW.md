# G4 candidate and CLM shadow trial

## CLM reflex adapter

`toddler.clm.HttpClmClient` uses the local CLM `/v1/systemone` endpoint and the
same four `noul` questions as JEV. `toddler.reflex_shadow.compare` records the
JEV and CLM decisions separately. Only the JEV decision is the existing reflex
decision; CLM has no actuator path. Both clients receive the same state, while
hard sensor limits are checked before either request. Missing, late, malformed,
NaN or infinite values stop the hypothetical reflex. The fixed `0.05` margin is
an uncalibrated comparison convention, not a physical safety guarantee.

The CLM implementation is pinned for a local pilot to upstream commit
`d5f9ef0fd9bde185df0ceaad4f4ecc6cfe8c34f6`, Qwen encoder
`Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`, and
CLM head `Contrastive-LM/CLM-v0.1-8B` revision
`e939398d4556fcd9400c76fa8c5a513202f42b0a`. Keep model weights outside
git. A local GPU lease and preflight are required before starting vLLM. On this
host a free V100 has more headroom than a 20 GB Ada for the 8B encoder.

The 9 October 2026 real pilot used 1Cat-vLLM 1.5.0 in pooling mode with FP16
on one V100 32 GB, the CLM head on CPU, and no CLM vector cache. The process
occupied about 24,583 MiB of GPU memory. One first cold four-question request
took 400 ms wall time. Twelve different synthetic states took 87–154 ms
(median 98 ms); twelve repeats of one state took 4.8–9.5 ms (median 5.1 ms).
The Toddler client completed one warmed request in 11.7 ms. These tiny samples
are transport timings, not a p99 guarantee or a risk-quality assessment.
Fresh states exceed the 20 ms reflex budget and must yield STOP. The JEV server
was reachable but returned an HTTP error for the paired pilot, so no
disagreement or quality result is claimed. The default vLLM 0.29 environment
failed on V100 because its PyTorch build lacked SM70; the separate 1Cat
environment worked.

Reproduce the pilot only after the shared GPU preflight and lease are granted;
the tested encoder command was:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
  /media/knight2/EDS2/envs/1cat-vllm-1.5.0/bin/vllm serve \
  /media/knight2/claude-data/knight1/models/qwen3-8b-clm \
  --served-model-name qwen3-8b --runner pooling --convert embed --dtype half \
  --max-model-len 2048 --gpu-memory-utilization 0.75 --enforce-eager \
  --host 127.0.0.1 --port 8091
PYTHONPATH=/media/knight2/claude-data/knight1/models/clm-source/src \
  python3 -m clm.server --host 127.0.0.1 --port 8700 \
  --emb-url http://127.0.0.1:8091/v1/embeddings --emb-model qwen3-8b \
  --ckpt /media/knight2/claude-data/knight1/models/clm-v0.1-8b/CLM_v0.1-8B.pt \
  --device cpu --action-cache 0 --no-download --no-ui
```

These are foreground pilot processes, not permanent services. Stop both and
release the lease when finished.

With the CLM and JEV endpoints running, create private JSONL rows of
`{"state": {...}, "sensors": {"force_n": 10, "speed_m_s": 0.1,
"person_distance_m": 2}}`, then run:

```bash
python3 scripts/compare_clm_reflex.py private-scenes.jsonl private-results.jsonl
```

The CLI checks `/health`, rejects a mock/unhealthy CLM backend, and writes only
decisions, error types and wall-clock timings to a mode-0600 file. It reports
p50/p95/p99 and disagreement counts. A missing JEV backend makes the paired
result incomplete. Synthetic states check transport only; independently labeled
hazard states are required before any safety claim or active use. Any future
active path must satisfy an end-to-end 20 ms deadline, not merely a socket
timeout.

## G4 curriculum and Teacher

The public training catalog includes arithmetic, English, Dutch, written
Cantonese, history, geography, trivia, culture, manners, puzzles, reasoning and
social-emotional questions. Its six Montessori-inspired areas are practical
life, sensorial, language, mathematics, cultural studies, and grace/courtesy.
MiniGrid `unlock` and `unlockpickup` are separate navigation skills. This is a
training taxonomy, not an IQ/EQ score or a Montessori certification.

`teacher.g4_nightly` trains a small cognitive answer-ranker on answer-keyed
practice items and inherits a verified G2 navigation policy. A simulator oracle
demonstrates `unlockpickup` on training maps; the student imitates actions using
only egocentric observations. The G2 trunk remains frozen during imitation.
PPO continues in a separate trial, preserving the imitation candidate.
Artifacts and a hash-bearing manifest stay under
`~/.local/share/teacher/g4/`. The run never reads private evaluation seeds and
never promotes G4. The fixed public practice questions and development seeds
cannot establish generalization; an independent hidden, matched comparison to
G3 and its ancestors is needed for promotion.

The first short navigation smoke run (13,000 environment steps) completed on
9 October 2026. `unlockpickup` scored **0.0** for both the parent and candidate
on ten public development seeds. A later oracle imitation run used 128 training
maps and 2,319 actions. On 50 separate public development seeds, the imitation
candidate solved `unlockpickup` **45/50** (normalised mean 0.9117) versus **0/50**
for G2, while preserving `unlock` at **50/50** for both. The separate PPO trial
solved `unlockpickup` 12/50. These public measurements choose a research candidate,
not a promoted G4. Cognitive training completed, but its development check shares
public question templates with training and is not a benchmark.

The next step is a five-child cohort matched to the five official
`G3-recombined` parents. Each child preserves its parent's eight other task
routes and inherited expert weights. A new `unlockpickup` specialist is routed
only if it gains at least 0.05 normalised mean on the fixed public development
set. An independent evaluator freezes the cohort and uses two fresh private
30-seed sets for a matched nine-task ancestor trial. See
[the G4 private protocol](learn/G4_ORACLE_TRIAL.md); neither the public route
selection nor the present cognitive template check is a promotion score.
The first five-child private trial improved the aggregate score but passed
only one of two required sets; [its audit result](learn/G4_ORACLE_RESULT_20261009.md)
keeps G4 unpromoted. The [separate cognitive pilot](learn/G4_COGNITIVE_HOLDOUT.md)
scored 22/54 on new private questions and is diagnostic only.
