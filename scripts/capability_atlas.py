"""Materialize and validate 10,000 explicitly unverified competency specs.

Each slot is a stable place for a sourced lesson, practical trial and assessment.
The generator never upgrades a slot to a demonstrated capability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from scripts.competency_specs import BRIEFS, PRACTICE, SETTING, competency_spec, spec_digest
except ModuleNotFoundError:  # direct `python scripts/capability_atlas.py`
    from competency_specs import BRIEFS, PRACTICE, SETTING, competency_spec, spec_digest

ROOT = Path(__file__).resolve().parents[1] / "knowledge" / "capabilities"

TOPICS = {
    "neuroscience": (
        "synaptic_plasticity", "metaplasticity", "neuromodulation", "hippocampal_replay",
        "systems_consolidation", "predictive_processing", "attention_and_control",
        "sensorimotor_learning", "developmental_learning", "neural_evidence_limits"),
    "learning_theory": (
        "generalization", "bias_variance", "optimization", "representation_learning",
        "transfer_learning", "meta_learning", "active_learning", "curriculum_learning",
        "causal_identification", "uncertainty_calibration"),
    "neural_networks": (
        "perceptrons_and_mlp", "convolutions", "attention_and_transformers",
        "recurrent_networks", "optimization_dynamics", "normalization",
        "parameter_efficient_adaptation", "sparse_experts", "plasticity_diagnostics",
        "model_compression"),
    "reinforcement_learning": (
        "mdps", "policy_gradients", "actor_critic", "ppo", "exploration",
        "imitation_learning", "offline_rl", "model_based_rl", "hierarchical_rl",
        "reward_design"),
    "continual_learning": (
        "catastrophic_forgetting", "experience_replay", "ewc", "progressive_networks",
        "adapter_isolation", "continual_backprop", "task_boundary_detection",
        "forward_transfer", "backward_transfer", "continual_rl_protocols"),
    "memory_systems": (
        "episodic_memory", "semantic_memory", "retrieval", "memory_provenance",
        "replay_sampling", "consolidation", "forgetting_policies", "working_memory",
        "long_context", "memory_evaluation"),
    "reasoning_and_language": (
        "symbolic_reasoning", "numeracy", "multilingual_learning", "program_synthesis",
        "tool_use", "planning", "self_correction", "grounded_dialogue",
        "preference_learning", "answer_verification"),
    "multimodal_agents": (
        "visual_perception", "speech_perception", "vision_language_alignment",
        "gui_grounding", "action_selection", "world_models", "robotic_control",
        "game_learning", "multi_agent_coordination", "human_feedback"),
    "evaluation_and_causality": (
        "measurement_design", "anti_contamination", "paired_trials", "sample_size",
        "ablation_studies", "calibration", "robustness", "energy_accounting",
        "lineage_auditing", "reproducibility"),
    "systems_and_safety": (
        "inference_routing", "gpu_scheduling", "model_serving", "latency_budgets",
        "data_governance", "fail_safe_control", "sandboxed_learning",
        "observability", "distributed_trials", "deployment_gates"),
}

PRACTICES = (
    "define", "explain", "derive", "implement", "reproduce", "diagnose",
    "compare", "design", "validate", "teach",
)
SETTINGS = (
    "worked_example", "simulated_task", "field_data", "few_shot",
    "task_sequence", "distribution_shift", "compute_budget", "human_review",
    "independent_holdout", "real_time",
)


def slots():
    if len(TOPICS) != 10 or len(PRACTICES) != 10 or len(SETTINGS) != 10:
        raise ValueError("the frozen atlas axes must have ten entries each")
    topic_keys = {f"{domain}/{topic}" for domain, topics in TOPICS.items() for topic in topics}
    if topic_keys != set(BRIEFS) or set(PRACTICES) != set(PRACTICE) or set(SETTINGS) != set(SETTING):
        raise ValueError("competency briefs do not cover exactly the frozen atlas axes")
    for domain, topics in TOPICS.items():
        if len(topics) != 10 or len(set(topics)) != 10:
            raise ValueError(f"{domain} must have ten distinct topics")
        for topic in topics:
            for practice in PRACTICES:
                for setting in SETTINGS:
                    capability_id = f"{domain}/{topic}/{practice}__{setting}"
                    spec = competency_spec(domain, topic, practice, setting)
                    yield capability_id, {
                        "schema": "toddler-capability-slot/v1",
                        "id": capability_id,
                        "domain": domain,
                        "topic": topic,
                        "practice": practice,
                        "setting": setting,
                        "target": spec["outcome"],
                        "competency": spec,
                        "status": "scaffold",
                        "evidence": {"primary_sources": [], "learning_material": [],
                                     "practical_artifacts": [], "assessment": []},
                        "mastery_claim": False,
                    }


def generate(root: Path = ROOT) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    index = []
    competency_counts = {"generated_unreviewed": 0, "editorial_reviewed": 0}
    for capability_id, record in slots():
        folder = root.joinpath(*capability_id.split("/"))
        folder.mkdir(parents=True, exist_ok=True)
        metadata = folder / "capability.json"
        if metadata.exists():
            actual = json.loads(metadata.read_text())
            if actual.get("id") != capability_id:
                raise ValueError(f"ID mismatch: {metadata}")
            current = actual.get("competency") or {}
            if current.get("status") == "editorial_reviewed":
                pass  # never overwrite a human-reviewed specification
            elif current.get("status") not in (None, "generated_unreviewed"):
                raise ValueError(f"unknown competency review status: {metadata}")
            elif current != record["competency"] or actual.get("target") != record["target"]:
                actual["competency"] = record["competency"]
                actual["target"] = record["target"]
                metadata.write_text(json.dumps(actual, ensure_ascii=False, indent=2) + "\n")
        else:
            actual = record
            metadata.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        competency_counts[actual["competency"]["status"]] += 1
        index.append(capability_id)
    if len(index) != 10_000 or len(set(index)) != 10_000:
        raise ValueError("atlas must contain exactly 10,000 unique slots")
    payload = "\n".join(index) + "\n"
    (root.parent / "CAPABILITY_INDEX.txt").write_text(payload)
    result = {"schema": "toddler-capability-atlas/v1", "slots": len(index),
              "domains": len(TOPICS), "topics": sum(map(len, TOPICS.values())),
              "index_sha256": hashlib.sha256(payload.encode()).hexdigest(),
              "mastery_claims": 0, "competency_specs": len(index),
              "competency_status_counts": competency_counts}
    (root.parent / "atlas.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def validate(root: Path = ROOT) -> dict:
    expected = dict(slots())
    actual_files = list(root.rglob("capability.json"))
    if len(actual_files) != len(expected):
        raise ValueError(f"expected 10,000 metadata files, found {len(actual_files)}")
    spec_digests = set()
    competency_counts = {"generated_unreviewed": 0, "editorial_reviewed": 0}
    for capability_id, intended in expected.items():
        metadata = root.joinpath(*capability_id.split("/")) / "capability.json"
        record = json.loads(metadata.read_text())
        if record.get("id") != capability_id or record.get("schema") != "toddler-capability-slot/v1":
            raise ValueError(f"invalid identity: {metadata}")
        spec = record.get("competency") or {}
        if spec.get("status") == "generated_unreviewed":
            if spec != intended["competency"] or record.get("target") != intended["target"]:
                raise ValueError(f"missing or stale competency specification: {metadata}")
        elif spec.get("status") == "editorial_reviewed":
            review = spec.get("review") or {}
            if (spec.get("schema") != "toddler-competency-spec/v1"
                    or record.get("target") != spec.get("outcome")
                    or not all(review.get(key) for key in ("reviewer", "reviewed_utc", "source_urls"))
                    or not isinstance(review["source_urls"], list)
                    or not all(isinstance(url, str) and url.startswith("https://") for url in review["source_urls"])
                    or len(spec.get("assessment", {}).get("criteria", [])) != 4):
                raise ValueError(f"invalid editorial review: {metadata}")
        else:
            raise ValueError(f"missing competency status: {metadata}")
        competency_counts[spec["status"]] += 1
        spec_digests.add(spec_digest(spec))
        if record.get("mastery_claim") is not False:
            raise ValueError(f"unreviewed slot claims mastery: {metadata}")
        if record.get("status") not in ("scaffold", "lesson_draft", "reviewed_lesson", "retired"):
            raise ValueError(f"unknown review status: {metadata}")
        if record["status"] == "lesson_draft" and (
                not (metadata.parent / "lesson.md").is_file()
                or not record.get("evidence", {}).get("primary_sources")
                or "lesson.md" not in record["evidence"].get("learning_material", [])):
            raise ValueError(f"draft lesson lacks a source or material: {metadata}")
    payload = "\n".join(expected) + "\n"
    summary = json.loads((root.parent / "atlas.json").read_text())
    if (len(spec_digests) != 10_000 or summary.get("competency_specs") != 10_000
            or summary.get("competency_status_counts") != competency_counts):
        raise ValueError("the atlas must contain 10,000 distinct competency specifications")
    if summary.get("index_sha256") != hashlib.sha256(payload.encode()).hexdigest():
        raise ValueError("atlas index digest differs from frozen taxonomy")
    if (root.parent / "CAPABILITY_INDEX.txt").read_text() != payload:
        raise ValueError("atlas index differs from frozen taxonomy")
    return {"validated_slots": len(expected), "validated_competencies": len(spec_digests),
            "index_sha256": summary["index_sha256"]}


def export_competencies(out: Path, root: Path = ROOT) -> dict:
    """Export all public specifications for Teacher; no private test content."""
    validate(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for capability_id, _ in slots():
            record = json.loads((root.joinpath(*capability_id.split("/")) / "capability.json").read_text())
            handle.write(json.dumps({"capability_id": capability_id, "lesson_status": record["status"],
                                     "competency": record["competency"], "mastery_claim": False},
                                    ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    temporary.replace(out)
    return {"competency_specs": count, "catalog": str(out),
            "catalog_sha256": hashlib.sha256(out.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("generate", "validate", "export"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, help="JSONL destination required for export")
    args = parser.parse_args()
    if args.action == "export":
        if args.out is None:
            parser.error("export requires --out")
        result = export_competencies(args.out, args.root)
    else:
        result = (generate if args.action == "generate" else validate)(args.root)
    print(json.dumps(result))
