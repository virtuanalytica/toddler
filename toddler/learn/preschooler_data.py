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


def prepare(rows: list[dict], *, protected_hashes: set[str], holdout_commit: str) -> tuple[list[dict], dict]:
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
        if not isinstance(row, dict) or row.get("schema") != SCHEMA:
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
        answer = _text(row.get("answer"), f"row {index} answer")
        model = row.get("model")
        if (not isinstance(model, dict) or model.get("access") != "local"
                or model.get("output_training_allowed") is not True):
            raise ValueError(f"row {index}: local model output and training rights are required")
        model_id = _text(model.get("id"), f"row {index} model.id")
        model_license = _text(model.get("license_id"), f"row {index} model.license_id")
        weights_hash = _sha(model.get("weights_sha256"), f"row {index} model.weights_sha256")
        if model_id in model_hashes and model_hashes[model_id] != weights_hash:
            raise ValueError(f"row {index}: same model id has multiple weight hashes")
        model_hashes[model_id] = weights_hash
        verification = row.get("verification")
        if not isinstance(verification, dict) or verification.get("passed") is not True:
            raise ValueError(f"row {index}: answer is not independently verified")
        method = verification.get("method")
        if method not in {"objective", "independent_expert"}:
            raise ValueError(f"row {index}: invalid verification method")
        _sha(verification.get("evidence_sha256"), f"row {index} verification.evidence_sha256")
        reviewer = _text(verification.get("reviewer_id"), f"row {index} reviewer_id")
        if reviewer == model_id:
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
                         "verification_method": method,
                         "verification_sha256": verification["evidence_sha256"]},
        })
    manifest = {"schema": "preschooler-sft-manifest/v1", "promotion_eligible": False,
                "holdout_commit_sha256": holdout_commit, "rows": len(accepted),
                "counts": dict(sorted(counts.items())), "model_weights_sha256": model_hashes,
                "source_families": dict(sorted(family_splits.items())),
                "warning": "Development data may select a candidate; only fresh independent holdout can promote it."}
    return accepted, manifest


def prepare_files(evidence: Path, protected: Path, destination: Path) -> dict:
    data = json.loads(protected.read_text(encoding="utf-8"))
    if data.get("schema") != "preschooler-protected-commit/v1":
        raise ValueError("invalid protected commitment manifest")
    rows = [json.loads(line) for line in evidence.read_text(encoding="utf-8").splitlines() if line.strip()]
    admitted, manifest = prepare(rows, protected_hashes=set(data["prompt_hashes"]),
                                 holdout_commit=data["holdout_commit_sha256"])
    manifest["evidence_file_sha256"] = hashlib.sha256(evidence.read_bytes()).hexdigest()
    destination.mkdir(parents=True, exist_ok=True)
    for split in ("train", "development"):
        output = destination / f"{split}.jsonl"
        output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                  for row in admitted if row["metadata"]["split"] == split), encoding="utf-8")
        manifest[f"{split}_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest
