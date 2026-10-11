"""Strict, local-only admission of verified examples for Preschooler SFT.

This module does not judge answers. Independent objective scorers or qualified
reviewers must supply evidence before a row can enter the training set.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

SCHEMA = "preschooler-evidence/v1"
MIXTURE_SCHEMA = "preschooler-evidence/v2"
DOMAINS = (
    "mathematics_logic", "physics", "chemistry", "software",
    "financial_data", "language_culture", "gamedev_ui", "agent_planning",
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().casefold())


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be SHA-256 hex")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value.strip()


def _model(model: object, label: str, model_hashes: dict[str, str]) -> tuple[str, str, str, str]:
    if (not isinstance(model, dict) or model.get("access") != "local"
            or model.get("output_training_allowed") is not True):
        raise ValueError(f"{label}: local model output and training rights are required")
    model_id = _text(model.get("id"), f"{label} id")
    license_id = _text(model.get("license_id"), f"{label} license_id")
    weights_hash = _sha(model.get("weights_sha256"), f"{label} weights_sha256")
    family = _text(model.get("family"), f"{label} family") if model.get("family") else model_id
    if model_id in model_hashes and model_hashes[model_id] != weights_hash:
        raise ValueError(f"{label}: same model id has multiple weight hashes")
    model_hashes[model_id] = weights_hash
    return model_id, license_id, weights_hash, family


def prepare(rows: list[dict], *, protected_hashes: set[str], holdout_commit: str,
            min_teacher_models: int = 1,
            teacher_roster: dict[str, dict] | None = None) -> tuple[list[dict], dict]:
    """Return training messages and a provenance manifest; reject whole batch on error."""
    _sha(holdout_commit, "holdout_commit")
    if not protected_hashes:
        raise ValueError("independent protected prompt hashes are required")
    for value in protected_hashes:
        _sha(value, "protected prompt hash")
    if not rows:
        raise ValueError("evidence set is empty")
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    family_splits: dict[str, str] = {}
    accepted: list[dict] = []
    counts: Counter[str] = Counter()
    model_hashes: dict[str, str] = {}
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or row.get("schema") not in {SCHEMA, MIXTURE_SCHEMA}:
            raise ValueError(f"row {index}: invalid schema")
        item_id = _text(row.get("item_id"), f"row {index} item_id")
        if item_id in seen_ids:
            raise ValueError(f"row {index}: duplicate item_id")
        seen_ids.add(item_id)
        split = row.get("split")
        if split not in {"train", "development"}:
            raise ValueError(f"row {index}: only train/development evidence is admitted")
        domain = row.get("domain")
        if domain not in DOMAINS:
            raise ValueError(f"row {index}: invalid domain")
        family = _text(row.get("source_family"), f"row {index} source_family")
        if family in family_splits and family_splits[family] != split:
            raise ValueError(f"row {index}: source family crosses train/development")
        family_splits[family] = split
        source = row.get("source")
        if (not isinstance(source, dict) or source.get("training_allowed") is not True
                or source.get("access") not in {"owned", "permissive"}):
            raise ValueError(f"row {index}: source rights not established")
        _text(source.get("license_id"), f"row {index} license_id")
        _sha(source.get("sha256"), f"row {index} source.sha256")
        prompt = _text(row.get("prompt"), f"row {index} prompt")
        prompt_hash = digest(normalized_prompt(prompt))
        if prompt_hash in protected_hashes:
            raise ValueError(f"row {index}: protected prompt overlap")
        if prompt_hash in seen_prompts:
            raise ValueError(f"row {index}: duplicate normalized prompt")
        seen_prompts.add(prompt_hash)
        contributors = []
        candidate_hashes = {}
        if row["schema"] == MIXTURE_SCHEMA:
            candidates = row.get("candidates")
            if not isinstance(candidates, list) or len(candidates) < 2:
                raise ValueError(f"row {index}: at least two candidates required")
            candidate_models, candidate_families = set(), set()
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    raise ValueError(f"row {index}: invalid candidate")
                candidate_id = _text(candidate.get("candidate_id"), f"row {index} candidate_id")
                if candidate_id in candidate_hashes:
                    raise ValueError(f"row {index}: duplicate candidate_id")
                candidate_answer = _text(candidate.get("answer"), f"row {index} candidate answer")
                model_id, license_id, weights_hash, model_family = _model(
                    candidate.get("model"), f"row {index} candidate {candidate_id}", model_hashes)
                revision = _text(candidate["model"].get("revision"),
                                 f"row {index} candidate {candidate_id} revision")
                if not re.fullmatch(r"[0-9a-f]{40}", revision):
                    raise ValueError(f"row {index}: candidate revision must be a pinned commit")
                if teacher_roster is not None:
                    pinned = teacher_roster.get(model_id)
                    if (pinned is None or pinned["revision"] != revision
                            or pinned["license_id"] != license_id
                            or pinned["family"] != model_family
                            or pinned.get("output_training_rights") != "approved"):
                        raise ValueError(f"row {index}: candidate differs from rights-approved roster")
                if model_id in candidate_models:
                    raise ValueError(f"row {index}: duplicate model candidate")
                candidate_models.add(model_id)
                candidate_families.add(model_family)
                candidate_hashes[candidate_id] = digest(candidate_answer)
                contributors.append({"candidate_id": candidate_id, "model_id": model_id,
                                     "model_family": model_family, "model_license_id": license_id,
                                     "model_revision": revision,
                                     "model_weights_sha256": weights_hash,
                                     "answer_sha256": candidate_hashes[candidate_id]})
            if len(candidate_families) < 2:
                raise ValueError(f"row {index}: candidates need two independent model families")
            selection = row.get("selection")
            if not isinstance(selection, dict):
                raise ValueError(f"row {index}: selection missing")
            answer = _text(selection.get("answer"), f"row {index} selected answer")
            selected_ids = selection.get("contributor_ids")
            if (not isinstance(selected_ids, list) or not selected_ids
                    or len(set(selected_ids)) != len(selected_ids)
                    or not set(selected_ids) <= set(candidate_hashes)):
                raise ValueError(f"row {index}: invalid selection contributor_ids")
            selection_method = selection.get("method")
            if selection_method not in {"chosen", "verified_synthesis"}:
                raise ValueError(f"row {index}: invalid selection method")
            if selection_method == "chosen" and (len(selected_ids) != 1 or
                    digest(answer) != candidate_hashes[selected_ids[0]]):
                raise ValueError(f"row {index}: chosen answer differs from its candidate")
            selected_models = [c for c in contributors if c["candidate_id"] in selected_ids]
            if selection_method == "verified_synthesis" and len(selected_models) < 2:
                raise ValueError(f"row {index}: synthesis requires at least two contributors")
            model_id = selected_models[0]["model_id"]
            model_license = selected_models[0]["model_license_id"]
            weights_hash = selected_models[0]["model_weights_sha256"]
        else:
            answer = _text(row.get("answer"), f"row {index} answer")
            model_id, model_license, weights_hash, _ = _model(row.get("model"),
                                                               f"row {index} model", model_hashes)
            selected_ids = ["legacy-single"]
            selection_method = "chosen"
            selected_models = [{"candidate_id": "legacy-single", "model_id": model_id,
                                "model_license_id": model_license,
                                "model_weights_sha256": weights_hash,
                                "answer_sha256": digest(answer)}]
        verification = row.get("verification")
        if not isinstance(verification, dict) or verification.get("passed") is not True:
            raise ValueError(f"row {index}: answer is not independently verified")
        method = verification.get("method")
        if method not in {"objective", "independent_expert"}:
            raise ValueError(f"row {index}: invalid verification method")
        _sha(verification.get("evidence_sha256"), f"row {index} verification.evidence_sha256")
        reviewer = _text(verification.get("reviewer_id"), f"row {index} reviewer_id")
        if reviewer in {c["model_id"] for c in contributors or selected_models}:
            raise ValueError(f"row {index}: model cannot review its own answer")
        overlap = row.get("overlap_audit")
        if not isinstance(overlap, dict) or overlap.get("status") != "passed":
            raise ValueError(f"row {index}: semantic/source overlap audit absent or failed")
        _sha(overlap.get("report_sha256"), f"row {index} overlap report")
        counts[f"{split}/{domain}"] += 1
        accepted.append({
            "messages": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": answer}],
            "metadata": {"item_id": item_id, "domain": domain, "split": split,
                         "source_family": family, "source_sha256": source["sha256"],
                         "license_id": source["license_id"], "model_id": model_id,
                         "model_license_id": model_license,
                         "model_weights_sha256": weights_hash, "prompt_sha256": prompt_hash,
                         "evidence_schema": row["schema"],
                         "candidate_models": contributors or selected_models,
                         "selected_candidate_ids": selected_ids,
                         "selection_method": selection_method,
                         "verification_method": method,
                         "verification_sha256": verification["evidence_sha256"]},
        })
    if len(model_hashes) < min_teacher_models:
        raise ValueError(f"only {len(model_hashes)} teacher models; require {min_teacher_models}")
    manifest = {"schema": "preschooler-sft-manifest/v1", "promotion_eligible": False,
                "holdout_commit_sha256": holdout_commit, "rows": len(accepted),
                "counts": dict(sorted(counts.items())), "model_weights_sha256": model_hashes,
                "source_families": dict(sorted(family_splits.items())),
                "warning": "Development data may select a candidate; only fresh independent holdout can promote it."}
    return accepted, manifest


def prepare_files(evidence: Path, protected: Path, destination: Path,
                  min_teacher_models: int = 1, roster_path: Path | None = None,
                  require_mixture: bool = False) -> dict:
    data = json.loads(protected.read_text(encoding="utf-8"))
    if data.get("schema") != "preschooler-protected-commit/v1":
        raise ValueError("invalid protected commitment manifest")
    rows = [json.loads(line) for line in evidence.read_text(encoding="utf-8").splitlines() if line.strip()]
    if require_mixture and any(row.get("schema") != MIXTURE_SCHEMA for row in rows):
        raise ValueError("multi-teacher export requires v2 evidence for every item")
    roster = None
    if roster_path is not None:
        roster_data = json.loads(roster_path.read_text(encoding="utf-8"))
        if roster_data.get("schema") != "preschooler-teacher-roster/v1":
            raise ValueError("invalid teacher roster")
        roster = {model["alias"]: model for model in roster_data["models"]}
    admitted, manifest = prepare(rows, protected_hashes=set(data["prompt_hashes"]),
                                 holdout_commit=data["holdout_commit_sha256"],
                                 min_teacher_models=min_teacher_models, teacher_roster=roster)
    manifest["evidence_file_sha256"] = hashlib.sha256(evidence.read_bytes()).hexdigest()
    destination.mkdir(parents=True, exist_ok=True)
    for split in ("train", "development"):
        output = destination / f"{split}.jsonl"
        output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                  for row in admitted if row["metadata"]["split"] == split), encoding="utf-8")
        manifest[f"{split}_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest
