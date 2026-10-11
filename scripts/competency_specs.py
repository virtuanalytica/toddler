"""Deterministic, assessable curriculum specifications for the frozen atlas.

These are authored task briefs, not lessons, answer keys or claims of mastery.
The 100 topic kernels combine with ten different professional practices and ten
operating settings. Each result states what to make, measure and falsify.
"""

from __future__ import annotations

import hashlib
import json


# topic | concrete artefact | observable measure | characteristic failure
_TOPIC_BRIEFS = """
neuroscience/synaptic_plasticity|a timing-dependent weight-update trace|change in synaptic efficacy by spike interval|confusing correlation with a causal learning rule
neuroscience/metaplasticity|a two-timescale plasticity simulation|adaptation of the learning threshold after prior activity|treating the threshold as fixed
neuroscience/neuromodulation|a reward-gated update simulation|credit assignment under delayed reward|equating dopamine with a scalar reward
neuroscience/hippocampal_replay|a replay schedule with online and offline phases|retention after delayed reactivation|claiming a biological mechanism from an engineering analogy
neuroscience/systems_consolidation|a fast-store to slow-store transfer model|retention after transfer and interference|losing source episodes during abstraction
neuroscience/predictive_processing|a hierarchical prediction-error model|calibrated prediction error across levels|assuming one neural implementation is established
neuroscience/attention_and_control|a selective-attention task with distractors|target accuracy and control cost|mistaking arousal for selective attention
neuroscience/sensorimotor_learning|an action-error adaptation trace|recovery after perturbation and washout|ignoring proprioceptive feedback
neuroscience/developmental_learning|a staged acquisition curriculum|transfer to the next developmental stage|projecting human ages onto model stages
neuroscience/neural_evidence_limits|an evidence-to-claim audit table|number of supported and unsupported mechanism claims|inferring cognition directly from an fMRI correlate
learning_theory/generalization|a train-test gap study|out-of-distribution error with uncertainty interval|tuning on the test partition
learning_theory/bias_variance|a model-complexity sweep|bias and variance estimated across resamples|using one seed to infer variance
learning_theory/optimization|a controlled optimizer comparison|loss and stability per compute unit|changing schedule and optimizer together
learning_theory/representation_learning|a probe and intervention on learned features|linear probe transfer and feature robustness|equating probe score with causal use
learning_theory/transfer_learning|a source-to-target adaptation study|target sample efficiency and negative transfer|using overlapping source and target items
learning_theory/meta_learning|an inner-loop outer-loop adaptation trial|new-task adaptation after fixed shots|leaking meta-test tasks into meta-training
learning_theory/active_learning|a query-selection simulation|labels needed to reach fixed quality|sampling from the hidden evaluation pool
learning_theory/curriculum_learning|an ordered-versus-shuffled training trial|learning-curve area and old-task retention|claiming curriculum gain without equal compute
learning_theory/causal_identification|a causal graph and intervention plan|identifiability under stated assumptions|controlling for a collider
learning_theory/uncertainty_calibration|a confidence-reliability study|expected calibration error and coverage|reporting confidence without calibration
neural_networks/perceptrons_and_mlp|a small multilayer network with gradient check|gradient error and held-out task accuracy|mistaking memorization for generalization
neural_networks/convolutions|a translation-sensitive feature extractor|accuracy under translations and parameter count|assuming convolution yields full invariance
neural_networks/attention_and_transformers|a masked-attention implementation|mask correctness and sequence scaling|leaking future tokens through the mask
neural_networks/recurrent_networks|a recurrent state-tracking model|long-horizon recall and gradient stability|resetting hidden state at the wrong boundary
neural_networks/optimization_dynamics|a loss-landscape and update trace|gradient norms and convergence variability|misreading noisy loss as convergence
neural_networks/normalization|a normalization ablation|training stability and activation statistics|mixing training and inference statistics
neural_networks/parameter_efficient_adaptation|a frozen-base adapter trial|quality per trainable parameter and memory|updating supposedly frozen weights
neural_networks/sparse_experts|a gated-expert routing study|expert load balance and token quality|expert collapse behind a high average score
neural_networks/plasticity_diagnostics|a dormant-unit and gradient-flow monitor|active-unit fraction across task switches|attributing every plateau to plasticity loss
neural_networks/model_compression|a quantization or pruning comparison|quality, latency and energy at matched tasks|using incompatible decode settings
reinforcement_learning/mdps|a finite-state transition and reward model|Bellman residual and policy return|violating the Markov assumption silently
reinforcement_learning/policy_gradients|a sampled policy-gradient estimator|return and gradient variance across seeds|reporting a single lucky rollout
reinforcement_learning/actor_critic|an actor-critic training trace|return, value error and stability|bootstrapping from terminal states
reinforcement_learning/ppo|a clipped-ratio policy update|return, KL drift and clipping frequency|treating clipping as a safety guarantee
reinforcement_learning/exploration|an exploration-policy comparison|coverage and regret under fixed steps|rewarding novelty without task progress
reinforcement_learning/imitation_learning|a demonstrator-to-policy cloning trial|success on unseen states and compounding error|evaluating only on demonstration trajectories
reinforcement_learning/offline_rl|an offline policy evaluation study|estimated versus observed return on supported actions|extrapolating beyond dataset support
reinforcement_learning/model_based_rl|a learned dynamics and planning loop|rollout error and return by planning horizon|trusting long imagined rollouts
reinforcement_learning/hierarchical_rl|a goal-option controller|task completion and option reuse|unreliable option termination
reinforcement_learning/reward_design|a reward-specification audit|goal success and proxy exploitation rate|reward hacking hidden by mean return
continual_learning/catastrophic_forgetting|a sequential-task retention matrix|old-task loss after each new task|reporting only final-task accuracy
continual_learning/experience_replay|a provenance-bound replay buffer|retention per byte and sample age|replaying evaluation examples
continual_learning/ewc|a Fisher-weighted parameter penalty|old-task retention versus new-task learning|assuming diagonal Fisher is exact
continual_learning/progressive_networks|a frozen-column lateral-transfer model|forward transfer per added parameter|growth without a memory budget
continual_learning/adapter_isolation|a task-specific adapter registry|cross-task interference and adapter cost|loading the wrong adapter at inference
continual_learning/continual_backprop|a selective dormant-unit reset trial|plasticity recovery and retained skills|resetting useful weights without rollback
continual_learning/task_boundary_detection|a shift-triggered task detector|boundary precision, recall and detection delay|firing on harmless input noise
continual_learning/forward_transfer|a new-task warm-start experiment|initial new-task performance versus scratch|counting test-task exposure as transfer
continual_learning/backward_transfer|a before-and-after old-task matrix|old-task improvement and regressions|hiding regressions in an average
continual_learning/continual_rl_protocols|a changing-environment RL schedule|return, regret and retention over task order|reusing a fixed task order as the holdout
memory_systems/episodic_memory|a timestamped event store and recall query|event precision and temporal ordering|inventing missing events
memory_systems/semantic_memory|a fact extraction and update ledger|fact consistency after updates|silently overwriting contradictory sources
memory_systems/retrieval|a query-to-evidence retriever|recall at k and answer support|ranking a familiar but irrelevant document
memory_systems/memory_provenance|a source-linked memory record|traceable claims and revocation completeness|orphaned facts after source deletion
memory_systems/replay_sampling|a replay selection policy|retention per replay token and class coverage|oversampling easy recent examples
memory_systems/consolidation|an episode-to-summary pipeline|factual retention and compression ratio|compressing away rare exceptions
memory_systems/forgetting_policies|a retention-and-deletion policy|correct deletion and downstream purge|keeping sensitive data in derived caches
memory_systems/working_memory|a bounded scratchpad controller|task success under a fixed state budget|carrying stale state into a new task
memory_systems/long_context|a long-document evidence task|evidence recall by position and context cost|assuming more context means better recall
memory_systems/memory_evaluation|a dated memory test with corrections|precision, temporal validity and abstention|grading against stale ground truth
reasoning_and_language/symbolic_reasoning|a rule-based inference trace|valid conclusions and contradiction rate|skipping an unstated premise
reasoning_and_language/numeracy|a units-aware calculation worksheet|exact answer rate and unit correctness|accepting a plausible wrong magnitude
reasoning_and_language/multilingual_learning|a cross-language meaning-preservation task|semantic equivalence and locale errors|treating translation fluency as factual accuracy
reasoning_and_language/program_synthesis|a specification-to-program trial|hidden-test pass rate and complexity|overfitting visible examples
reasoning_and_language/tool_use|a tool-call plan with observable execution|task completion and invalid-call rate|claiming success without checking the tool result
reasoning_and_language/planning|a dependency-aware action plan|goal completion, cost and replans|following a stale plan after new evidence
reasoning_and_language/self_correction|an error-detection and repair trace|net correction gain and new-error rate|changing a correct answer without evidence
reasoning_and_language/grounded_dialogue|an evidence-cited dialogue response|claim support and appropriate abstention|citing a source that does not support the claim
reasoning_and_language/preference_learning|a pairwise preference dataset audit|annotator agreement and ranking generalization|treating style preference as factual truth
reasoning_and_language/answer_verification|a claim-by-claim verification record|true-positive error detection and false alarms|rubber-stamping fluent answers
multimodal_agents/visual_perception|an image-to-object inventory|detection precision and missed objects|hallucinating occluded detail
multimodal_agents/speech_perception|a speech-to-intent transcript|word error and intent accuracy by noise level|inventing speech in silence
multimodal_agents/vision_language_alignment|a text-to-region grounding record|region match and caption faithfulness|using language priors over image evidence
multimodal_agents/gui_grounding|a screenshot-to-control action trace|correct target and safe click rate|acting on stale coordinates
multimodal_agents/action_selection|a state-action decision log|goal progress and avoidable mistakes|choosing an irreversible action too early
multimodal_agents/world_models|a state-transition predictor|rollout error by horizon|planning past the model's reliable horizon
multimodal_agents/robotic_control|a bounded robot-control trajectory|goal completion and collision count|testing unsafe motion on live hardware
multimodal_agents/game_learning|a novel-game rules and strategy notebook|learning speed and win rate on unseen levels|using hidden level scripts as training data
multimodal_agents/multi_agent_coordination|a role-and-message coordination trace|team completion and message overhead|duplicate work from ambiguous ownership
multimodal_agents/human_feedback|a feedback-to-policy update record|improvement and disagreement rate|assuming one person's preference is universal
evaluation_and_causality/measurement_design|a construct-to-metric specification|inter-rater reliability and construct coverage|optimizing an easy proxy
evaluation_and_causality/anti_contamination|a leakage threat model and item lineage ledger|detected overlaps and sealed-set integrity|training on a supposedly private evaluation item
evaluation_and_causality/paired_trials|a paired baseline-candidate experiment|paired effect and confidence interval|comparing different prompt or decode settings
evaluation_and_causality/sample_size|a prospective power and precision worksheet|required sample size under stated effect|claiming significance after repeated peeking
evaluation_and_causality/ablation_studies|a one-factor intervention matrix|incremental effect with uncertainty|changing several components at once
evaluation_and_causality/calibration|a reliability and abstention curve|calibration error and risk-coverage tradeoff|calling a confident model correct
evaluation_and_causality/robustness|a perturbation stress battery|worst-group score and failure rate|reporting only aggregate mean
evaluation_and_causality/energy_accounting|a metered energy-per-answer ledger|GPU-board Wh and system Wh separately|presenting GPU-board energy as total power
evaluation_and_causality/lineage_auditing|a model-data-run provenance chain|complete traceability to hashes and versions|missing a training-data or adapter version
evaluation_and_causality/reproducibility|an independent rerun package|agreement within preregistered tolerance|omitting seed, environment or data snapshot
systems_and_safety/inference_routing|a workload-to-model routing policy|quality, latency and cost per request class|selecting on hidden test answers
systems_and_safety/gpu_scheduling|a GPU lease and cleanup protocol|utilization, contention and recovery time|leaving stale processes in allocated VRAM
systems_and_safety/model_serving|a load-testable inference endpoint|throughput, error rate and tail latency|benchmarking only unloaded service
systems_and_safety/latency_budgets|an end-to-end latency allocation|p95 response time by stage|hiding queue time outside the measurement
systems_and_safety/data_governance|a data classification and retention register|policy compliance and deletion verification|sending restricted material to an external API
systems_and_safety/fail_safe_control|a safe-stop and rollback state machine|time to safe state and recovery success|assuming a timeout means completion
systems_and_safety/sandboxed_learning|an isolated candidate-training environment|containment and artifact provenance|allowing training code to read private tests
systems_and_safety/observability|a metric-trace-alert contract|detection delay and false-alert rate|logging sensitive prompts into dashboards
systems_and_safety/distributed_trials|a federated trial manifest|reproducible merges and conflict resolution|mixing incomparable hardware results
systems_and_safety/deployment_gates|a versioned promotion and rollback gate|false promotion rate and rollback readiness|promoting without independent evidence
""".strip()


def _briefs() -> dict[str, tuple[str, str, str]]:
    rows = {}
    for line in _TOPIC_BRIEFS.splitlines():
        key, artifact, measure, failure = line.split("|", 3)
        if key in rows:
            raise ValueError(f"duplicate topic brief: {key}")
        rows[key] = (artifact, measure, failure)
    return rows


BRIEFS = _briefs()

PRACTICE = {
    "define": ("Specify operational terms and boundaries for", "a glossary with a positive and a negative case", "the definitions separate valid from invalid cases"),
    "explain": ("Explain the mechanism and its assumptions behind", "an annotated causal or computational diagram", "each claimed link has an assumption and a possible falsifier"),
    "derive": ("Derive a formal or quantitative account of", "a stepwise derivation with units or invariants", "the derivation reproduces a checked special case"),
    "implement": ("Build a runnable demonstration of", "executable code or a runnable protocol", "the artifact passes a new functional case and a failure case"),
    "reproduce": ("Reproduce a stated result for", "a rerun manifest with data, version, seeds and commands", "the rerun agrees within a predeclared tolerance"),
    "diagnose": ("Locate a failure and test a causal remedy for", "a fault trace, hypothesis table and controlled repair", "the repair fixes the target without hiding a regression"),
    "compare": ("Compare two plausible methods for", "a paired comparison and tradeoff table", "same inputs and budgets yield a justified choice"),
    "design": ("Design a deployable workflow for", "a design, interfaces, failure paths and resource budget", "an independent reviewer can execute its acceptance tests"),
    "validate": ("Validate a claim about", "a preregistered test protocol and evidence record", "a held-out challenge supports or rejects the claim"),
    "teach": ("Teach another learner to reason about", "a lesson, coached exercise and blind learner transfer test", "the learner succeeds on a new case without the answer key"),
}

SETTING = {
    "worked_example": ("Use a small, fully specified worked case and then change one input.", "Include the solved case and the changed-case prediction."),
    "simulated_task": ("Use a seeded simulation with an explicit environment version.", "Include the simulator config, seed band and baseline run."),
    "field_data": ("Use a provenance-checked real dataset with consent and access rules.", "Include the data card, missingness check and permitted use."),
    "few_shot": ("Limit adaptation to a declared small example budget.", "Include the example count, selection rule and learning curve."),
    "task_sequence": ("Run at least three ordered tasks and revisit the first.", "Include per-task before-and-after measurements."),
    "distribution_shift": ("Fit on one distribution and evaluate on a separately documented shift.", "Include shift definition and in/out-distribution results."),
    "compute_budget": ("Fix a time, memory and energy budget before the run.", "Include resource log and quality per unit of compute."),
    "human_review": ("Use a reviewer who did not author the artifact or see hidden answers.", "Include reviewer notes, disagreement resolution and audit trail."),
    "independent_holdout": ("Have a separate evaluator hold unseen items and answer keys.", "Include item-set hash, evaluator identity and aggregate verdict only."),
    "real_time": ("Meet a declared latency and fail-safe limit under concurrent load.", "Include p95 latency, timeout behavior and rollback trace."),
}


def competency_spec(domain: str, topic: str, practice: str, setting: str) -> dict:
    """Return one stable task specification with no invented results or sources."""
    artifact, measure, failure = BRIEFS[f"{domain}/{topic}"]
    action, deliverable, action_check = PRACTICE[practice]
    constraint, setting_evidence = SETTING[setting]
    title = f"{practice.replace('_', ' ').capitalize()} {topic.replace('_', ' ')} — {setting.replace('_', ' ')}"
    target = f"{action} {topic.replace('_', ' ')} using {artifact}. {constraint}"
    spec = {
        "schema": "toddler-competency-spec/v1",
        "title": title,
        "outcome": target,
        "topic_kernel": {"artifact": artifact, "measure": measure, "failure_to_detect": failure},
        "practice_task": f"Produce {deliverable} grounded in {artifact}.",
        "setting_constraint": constraint,
        "required_evidence": [deliverable, setting_evidence,
                              f"Report {measure} with uncertainty or an explicit deterministic check.",
                              f"Construct a negative control for the failure mode: {failure}."],
        "assessment": {
            "mode": "independent_evaluator_new_case",
            "pass_rule": "all_four_criteria_required",
            "scoring": "Each criterion: 0 absent, 1 partial, 2 demonstrated; require 2 on all four.",
            "threshold_policy": "Register the metric threshold, baseline, task budget and evaluator before exposing new items.",
            "criteria": [
                f"Concept: correctly state the assumptions and limits of {topic.replace('_', ' ')}.",
                f"Practice: {action_check}.",
                f"Measurement: report {measure} against a matched baseline with reproducible inputs.",
                f"Critical check: test for {failure}; detect it or rule it out with evidence.",
            ],
        },
        "source_policy": "Reviewer must add a verified primary source before lesson status; no source is fabricated here.",
        "contamination_policy": "No private items, answers, seeds or reviewer-only prompts in this public specification.",
        "status": "generated_unreviewed",
    }
    return spec


def spec_digest(spec: dict) -> str:
    payload = json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
