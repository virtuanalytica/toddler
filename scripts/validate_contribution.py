"""Validate disclosure and local trial hashes for distributed Toddler PRs.

This is a PR integrity check, not the private promotion evaluator. A merged
submission remains a candidate until independent hidden trials pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
BLOCKED_FIELDS = {"seeds", "seed_values", "salt", "api_key", "private_key", "access_token", "password"}
PLACEHOLDERS = {"", "unknown", "n/a", "private", "tbd", "redacted"}


def _text(value, label: str) -> str:
    if not isinstance(value, str) or value.strip().lower() in PLACEHOLDERS:
        raise ValueError(f"{label} must disclose a concrete value")
    return value.strip()


def _hash(value, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    return value


def _no_private_values(value, label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in BLOCKED_FIELDS:
                raise ValueError(f"{label}: raw secrets or private seed values belong outside Git ({key})")
            _no_private_values(child, label)
    elif isinstance(value, list):
        for child in value:
            _no_private_values(child, label)


def _artifact(manifest: Path, root: Path, relative: str, expected: str) -> dict:
    rel = Path(_text(relative, "artifact path"))
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("trial artifacts must stay inside the submission")
    path = (manifest.parent / rel).resolve()
    if not path.is_relative_to(manifest.parent.resolve()) or not path.is_relative_to(root.resolve()):
        raise ValueError("trial artifact escapes the repository")
    if hashlib.sha256(path.read_bytes()).hexdigest() != _hash(expected, "trial artifact hash"):
        raise ValueError(f"trial artifact hash mismatch: {relative}")
    body = json.loads(path.read_text())
    _no_private_values(body, str(path))
    return body


def validate(path: Path, root: Path = REPO) -> dict:
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("submission manifest must be inside the repository")
    body = json.loads(path.read_text())
    _no_private_values(body, str(path))
    if body.get("schema") != "toddler-contribution/v1":
        raise ValueError("unsupported contribution schema")
    contributor, candidate, recipe = body["contributor"], body["candidate"], body["recipe"]
    _text(contributor["github"], "contributor.github")
    _text(contributor["name"], "contributor.name")
    _text(candidate["generation"], "candidate.generation")
    _text(candidate["id"], "candidate.id")
    _hash(candidate["weights_sha256"], "candidate.weights_sha256")
    if not isinstance(candidate.get("parent_refs"), list) or not candidate["parent_refs"]:
        raise ValueError("candidate needs registered parent refs")
    for parent in candidate["parent_refs"]:
        if "/" not in _text(parent, "candidate.parent_ref"):
            raise ValueError("parent refs must be generation/toddler_id")
    _text(recipe["summary"], "recipe.summary")
    if not isinstance(recipe.get("commands"), list) or not recipe["commands"]:
        raise ValueError("recipe needs reproducible commands")
    for command in recipe["commands"]:
        _text(command, "recipe.command")
    if not isinstance(recipe.get("environment"), dict) or not recipe["environment"]:
        raise ValueError("recipe needs environment versions")
    if not isinstance(recipe.get("data_sources"), list) or not recipe["data_sources"]:
        raise ValueError("recipe needs data provenance")
    for source in recipe["data_sources"]:
        _text(source["id"], "data source ID")
        _text(source["provenance"], "data source provenance")
        if source["visibility"] not in ("public", "private"):
            raise ValueError("data source visibility must be public or private")
    models = recipe["models_used"]
    if not isinstance(models, list) or (not models and recipe.get("no_external_models") is not True):
        raise ValueError("disclose every model or explicitly declare no external models")
    if models and recipe.get("no_external_models") is True:
        raise ValueError("models_used contradicts no_external_models")
    for model in models:
        for field in ("provider", "model_id", "revision", "role", "usage"):
            _text(model[field], f"model.{field}")
        if model["visibility"] not in ("public", "private"):
            raise ValueError("model visibility must be public or private")
    interactions = recipe["interactions"]
    if not isinstance(interactions, list):
        raise ValueError("interactions must be disclosed, possibly as an empty list")
    for interaction in interactions:
        _text(interaction["kind"], "interaction.kind")
        if not isinstance(interaction["count"], int) or interaction["count"] < 1:
            raise ValueError("interaction count must be positive")
        _hash(interaction["log_sha256"], "interaction.log_sha256")
    trials = body["trials"]
    if not isinstance(trials, list) or not trials:
        raise ValueError("at least one contributor trial is required")
    seen_sets = set()
    for trial in trials:
        if trial["kind"] not in ("contributor_public", "contributor_private"):
            raise ValueError("official private evaluation is recorded separately from contributor trials")
        seed_set = _text(trial["seed_set_id"], "trial.seed_set_id")
        if seed_set in seen_sets:
            raise ValueError("trial seed-set IDs must differ")
        seen_sets.add(seed_set)
        _text(trial["matched_control"], "trial.matched_control")
        if trial["eval_mode"] not in ("greedy", "sample"):
            raise ValueError("trial.eval_mode must be greedy or sample")
        protocol = _artifact(path, root, trial["protocol"], trial["protocol_sha256"])
        report = _artifact(path, root, trial["report"], trial["report_sha256"])
        if (protocol.get("seed_set_id") != seed_set or report.get("seed_set_id") != seed_set
                or report.get("protocol_sha256") != trial["protocol_sha256"]):
            raise ValueError("trial report, protocol and seed-set ID disagree")
    return {"candidate": candidate["generation"] + "/" + candidate["id"],
            "contributor": contributor["github"], "models_disclosed": len(models),
            "trials": len(trials), "promotion": "requires independent official hidden trials"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest", type=Path)
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()
    manifests = [args.manifest] if args.manifest else sorted((REPO / "submissions").glob("**/submission.json"))
    for path in manifests:
        print(json.dumps(validate(path)))
    print(f"validated {len(manifests)} contribution manifest(s)")


if __name__ == "__main__":
    main()
