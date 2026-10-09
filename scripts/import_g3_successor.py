"""Verify the frozen G3 successor; import it only after a signed human review.

The approval file is signed with ``ssh-keygen -Y sign -n toddler-lineage``.
The operator supplies an SSH allowed-signers file to ``apply``. ``check`` is
read-only and needs no approval. Both paths verify the complete evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import torch

from scripts.benchmark_routed_successor import verify_sources
from toddler.learn.generations import Registry
from toddler.learn.lineage import Inheritance, Ledger, code_version
from toddler.learn.routing import TaskExpertRouter

REPO = Path(__file__).resolve().parents[1]
REVIEW = REPO / "docs/learn/G3_RECOMBINED_REVIEW.json"
REPLICATION = Path("/media/knight2/EDS2/toddler-g3-recombined-replication-20261008/registry")
GENERATION = "G3-recombined"
CHILDREN = tuple(f"t{5000 + i}" for i in range(1, 6))
ORIGINAL_TASKS = ("cartpole", "acrobot", "empty5", "doorkey5")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(review_path: Path = REVIEW, replication_root: Path = REPLICATION,
            official_root: Path | None = None, trial_root: Path | None = None) -> dict:
    packet = json.loads(review_path.read_text())
    if packet.get("schema") != "toddler-lineage-review/v1" or packet.get("candidate_generation") != GENERATION:
        raise ValueError("wrong lineage review packet")
    if packet.get("decision") != "pending_human_review" or packet.get("approval") != {
            "reviewed_by": None, "approved_at": None, "signature": None}:
        raise ValueError("review packet has changed; use a fresh, independently checked import")
    if packet.get("current_official_generation") != "G2" or len(packet.get("confirmations", [])) != 2:
        raise ValueError("expected G2 and two independent confirmations")
    official_root = official_root or Path(packet["official_registry"])
    trial_root = trial_root or Path(packet["trial_registry"])
    ledger = Ledger(official_root)
    if ledger.verify() != (True, None) or (ledger.verdict("G2") or {}).get("verdict") != "survived":
        raise ValueError("official lineage is invalid or G2 did not survive")
    protocols, reports = [], []
    for confirmation in packet["confirmations"]:
        paths = {}
        for name in ("protocol", "report"):
            relative = Path(confirmation[name])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("review references a path outside the repository")
            path = REPO / relative
            if digest(path) != confirmation[f"{name}_sha256"]:
                raise ValueError(f"{name} hash differs from the review packet")
            paths[name] = json.loads(path.read_text())
        protocol, report = paths["protocol"], paths["report"]
        if report["protocol_sha256"] != confirmation["protocol_sha256"]:
            raise ValueError("report cites another protocol")
        if (report["secret_seed_commitment"] != protocol["secret_seed_commitment"]
                or report["secret_seed_commitment"] != confirmation["secret_seed_commitment"]):
            raise ValueError("seed commitment differs")
        if (not report["eligible_for_lineage_review"] or not report["beats_all_source_IQMs"]
                or not report["preserves_G1_tasks"]
                or not report["promotion_vs_G2"]["promote"]
                or not report["promotion_vs_random_control"]["promote"]
                or not confirmation["eligible_for_lineage_review"]):
            raise ValueError("a registered promotion condition failed")
        if (report["routes"] != protocol["candidate_routes"]
                or any(route[task] != "G2" for route in report["routes"].values() for task in ORIGINAL_TASKS)):
            raise ValueError("candidate route changed or old tasks were rerouted")
        if {row["id"]: row["weights_sha256"] for row in report["children"] if row["arm"] == "candidate"} != packet["candidate_weight_sha256"]:
            raise ValueError("candidate child hashes differ from review packet")
        protocols.append(protocol)
        reports.append(report)
    if protocols[0]["secret_seed_set_id"] == protocols[1]["secret_seed_set_id"]:
        raise ValueError("replication reused the first seed set")
    for key in ("source_weights", "candidate_routes", "random_control_routes", "promotion_rule"):
        if protocols[0][key] != protocols[1][key]:
            raise ValueError(f"replication changed {key}")
    source, trial, replica = Registry(official_root), Registry(trial_root), Registry(replication_root)
    for reg in (trial, replica):
        if {record.toddler_id for record in reg.generation(GENERATION)} != set(CHILDREN):
            raise ValueError("candidate registry must contain exactly the five reviewed children")
    for index, child_id in enumerate(CHILDREN, 1):
        parent_id = f"t{2000 + index}"
        expected = packet["candidate_weight_sha256"][child_id]
        for reg, protocol, confirmation in zip((trial, replica), protocols, packet["confirmations"]):
            net, record = reg.load(GENERATION, child_id)
            if not isinstance(net, TaskExpertRouter) or record.weights_sha256 != expected:
                raise ValueError(f"{child_id} has a different model or weight hash")
            if record.config["route"] != protocols[0]["candidate_routes"][parent_id]:
                raise ValueError(f"{child_id} has a different route")
            verify_sources(net, record, parent_id, source, protocol,
                           confirmation["protocol_sha256"])
            if set(record.parents) != {f"{g}/{parent_id if g.startswith('G2') else f't{3000 + index}'}"
                                       for g in protocols[0]["source_generations"]}:
                raise ValueError(f"{child_id} does not cite all source experts")
    existing = official_root / GENERATION
    if existing.exists():
        if {record.toddler_id for record in source.generation(GENERATION)} != set(CHILDREN):
            raise ValueError("official G3 registry has unreviewed children")
        for child_id in CHILDREN:
            _, record = source.load(GENERATION, child_id)
            if record.weights_sha256 != packet["candidate_weight_sha256"][child_id]:
                raise ValueError("official G3 weights differ from frozen candidate")
    verdict = ledger.verdict(GENERATION)
    if verdict and verdict["verdict"] != "survived":
        raise ValueError("official G3 has a conflicting selection verdict")
    return {"review_sha256": digest(review_path), "review": packet,
            "official_root": official_root, "trial_root": trial_root,
            "replication_root": replication_root, "report": reports[0],
            "replication_report": reports[1], "already_imported": existing.exists()}


def verify_approval(evidence: dict, approval_path: Path, signature_path: Path,
                    allowed_signers: Path) -> dict:
    raw = approval_path.read_bytes()
    approval = json.loads(raw)
    if (set(approval) != {"candidate_generation", "decision", "review_sha256", "reviewed_by", "approved_at"}
            or approval["candidate_generation"] != GENERATION or approval["decision"] != "approve"
            or approval["review_sha256"] != evidence["review_sha256"]):
        raise ValueError("approval does not authorize this exact review packet")
    reviewer = approval["reviewed_by"]
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("approval needs an identified reviewer")
    timestamp = datetime.fromisoformat(approval["approved_at"].replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("approval timestamp needs a timezone")
    proc = subprocess.run(["ssh-keygen", "-Y", "verify", "-f", str(allowed_signers),
                           "-I", reviewer, "-n", "toddler-lineage", "-s", str(signature_path)],
                          input=raw, capture_output=True)
    if proc.returncode:
        raise ValueError("review signature is not valid for the trusted signer identity")
    return {"reviewed_by": reviewer, "approved_at": approval["approved_at"],
            "approval_sha256": hashlib.sha256(raw).hexdigest(),
            "signature_sha256": digest(signature_path)}


def apply(evidence: dict, approval: dict) -> dict:
    root, trial_root = evidence["official_root"], evidence["trial_root"]
    target = root / GENERATION
    if not target.exists():
        staging = Path(tempfile.mkdtemp(prefix=".g3-import-", dir=root))
        try:
            shutil.copytree(trial_root / GENERATION, staging / GENERATION)
            for child_id, expected in evidence["review"]["candidate_weight_sha256"].items():
                _, record = Registry(staging).load(GENERATION, child_id)
                if record.weights_sha256 != expected:
                    raise ValueError("staged weights differ from reviewed weights")
            os.replace(staging / GENERATION, target)
        finally:
            shutil.rmtree(staging)
    ledger = Ledger(root)
    for child_id in CHILDREN:
        ref = f"{GENERATION}/{child_id}"
        net, record = Registry(root).load(GENERATION, child_id)
        if record.weights_sha256 != evidence["review"]["candidate_weight_sha256"][child_id]:
            raise ValueError("official weights changed during import")
        if ledger.birth(ref):
            if ledger.birth(ref)["weights_sha256"] != record.weights_sha256:
                raise ValueError("existing lineage birth has other weights")
            continue
        inherits = []
        for generation, expert in net.experts.items():
            parent = next(p for p in record.parents if p.startswith(generation + "/"))
            inherits.append(Inheritance(parent, ledger.weights_of(parent) or "",
                                        tuple("task:" + task for task, selected in net.route.items()
                                              if selected == generation)))
            if not inherits[-1].parent_weights_sha256:
                raise ValueError(f"source {parent} missing from official lineage")
        ledger.born(ref, record.weights_sha256, role="population", inherits=inherits,
                    code=code_version(), data={"protocol_sha256": evidence["review"]["confirmations"][0]["protocol_sha256"],
                                               "replication_protocol_sha256": evidence["review"]["confirmations"][1]["protocol_sha256"],
                                               "review_sha256": evidence["review_sha256"]},
                    budget={"new_environment_steps": 0, "method": "frozen-task-expert-route"},
                    hardware=record.hardware, software=record.software)
    prior = ledger.verdict(GENERATION)
    if prior:
        if prior["verdict"] != "survived" or prior["evidence"].get("approval_sha256") != approval["approval_sha256"]:
            raise ValueError("generation already has a different selection verdict")
    else:
        trial = evidence["report"]
        ledger.selected(GENERATION, "survived", {
            "review_sha256": evidence["review_sha256"], **approval,
            "trial_report_sha256": evidence["review"]["confirmations"][0]["report_sha256"],
            "replication_report_sha256": evidence["review"]["confirmations"][1]["report_sha256"],
            "promotion_vs_G2": trial["promotion_vs_G2"],
            "promotion_vs_random_control": trial["promotion_vs_random_control"],
            "aggregate_iqm": trial["secret"]["candidate"]["aggregate_iqm"],
            "limitations": evidence["review"]["known_limitations"],
        })
    if ledger.verify() != (True, None):
        raise ValueError("lineage chain failed verification after import")
    return {"generation": GENERATION, "verdict": ledger.verdict(GENERATION)["verdict"],
            "members": len(ledger.members(GENERATION)), "ledger_verified": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "apply"))
    parser.add_argument("--review", type=Path, default=REVIEW)
    parser.add_argument("--replication-root", type=Path, default=REPLICATION)
    parser.add_argument("--official-root", type=Path)
    parser.add_argument("--trial-root", type=Path)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--signature", type=Path)
    parser.add_argument("--allowed-signers", type=Path)
    args = parser.parse_args()
    evidence = inspect(args.review, args.replication_root, args.official_root, args.trial_root)
    if args.command == "check":
        print(json.dumps({"candidate": GENERATION, "ready_for_review": True,
                          "review_sha256": evidence["review_sha256"],
                          "already_imported": evidence["already_imported"]}, indent=2))
        return
    if not all((args.approval, args.signature, args.allowed_signers)):
        parser.error("apply needs --approval, --signature and --allowed-signers")
    approved = verify_approval(evidence, args.approval, args.signature, args.allowed_signers)
    print(json.dumps(apply(evidence, approved), indent=2))


if __name__ == "__main__":
    main()
