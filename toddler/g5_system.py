"""Load a G5 public prototype as one cognitive and navigation runtime.

Only owner-created, hash-checked local joblib artifacts may be loaded. The
private promotion evaluator must use a separate held-out family and verdict.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from toddler.g4_curriculum import PracticeItem
from toddler.g5_cognitive import G5CognitiveStudent
from toddler.learn.generations import Registry
from toddler.learn.routing import TaskExpertRouter


@dataclass
class G5Prototype:
    ref: str
    navigation: TaskExpertRouter
    cognitive: G5CognitiveStudent

    @classmethod
    def load(cls, manifest_path: Path, ref: str, *, parent_root: Path,
             parent_contract: Path) -> "G5Prototype":
        import joblib
        from scripts.g5_parent_contract import module_digest, verify_contract

        manifest_path = Path(manifest_path).resolve()
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("schema") != "toddler-g5-public-candidate/v1"
                or manifest.get("status") != "trained_public_candidate_unpromoted"
                or manifest.get("private_evaluation") != "not_started"
                or manifest.get("code", {}).get("dirty") is not False):
            raise ValueError("candidate has no clean, non-promoted G5 prototype manifest")
        if hashlib.sha256(Path(parent_contract).read_bytes()).hexdigest() != manifest.get("parent_contract_sha256"):
            raise ValueError("G5 parent contract bytes differ from candidate commitment")
        frozen = json.loads(Path(parent_contract).read_text())
        verify_contract(parent_root, frozen)
        matches = [row for row in manifest.get("children", []) if row.get("ref") == ref]
        if len(matches) != 1:
            raise ValueError("candidate child ref is missing or duplicated")
        row = matches[0]
        if not ref.startswith("G5-public-candidate/t700"):
            raise ValueError("unexpected public G5 candidate ID")
        parent_ref = row["navigation_parent"]
        generation, toddler_id = parent_ref.split("/", 1)
        if generation != "G4-search" or toddler_id[-1] != ref[-1]:
            raise ValueError("navigation parent is not a promoted G4 search child")
        frozen_parent = next((parent for parent in frozen["parents"] if parent["ref"] == parent_ref), None)
        if (frozen_parent is None or row["navigation_parent_weights_sha256"] != frozen_parent["weights_sha256"]
                or row["inherited_expert_sha256"] != frozen_parent["expert_sha256"]
                or row["inherited_route_sha256"] != frozen_parent["route_sha256"]):
            raise ValueError("candidate inheritance differs from frozen G4 parent contract")
        nav, record = Registry(parent_root).load(generation, toddler_id)
        if not isinstance(nav, TaskExpertRouter) or record.weights_sha256 != row["navigation_parent_weights_sha256"]:
            raise ValueError("navigation weights differ from candidate parent commitment")
        if {name: module_digest(expert) for name, expert in nav.experts.items()} != row["inherited_expert_sha256"]:
            raise ValueError("navigation expert tensors differ from candidate parent commitment")
        route_digest = hashlib.sha256(json.dumps(nav.route, sort_keys=True,
                                                 separators=(",", ":")).encode()).hexdigest()
        if route_digest != row["inherited_route_sha256"]:
            raise ValueError("navigation route differs from candidate parent commitment")
        cognitive_path = manifest_path.parent / "cognitive.joblib"
        digest = hashlib.sha256(cognitive_path.read_bytes()).hexdigest()
        if digest != row["cognitive_artifact_sha256"] or digest != manifest["cognitive_artifact_sha256"]:
            raise ValueError("cognitive artifact differs from candidate commitment")
        cognitive = joblib.load(cognitive_path)
        if not isinstance(cognitive, G5CognitiveStudent):
            raise ValueError("candidate cognitive artifact has the wrong type")
        return cls(ref, nav.eval(), cognitive)

    def answer(self, item: PracticeItem) -> tuple[str, float]:
        return self.cognitive.choose(item)

    def navigation_action(self, task: str, observation: torch.Tensor) -> int:
        if task not in self.navigation.route:
            raise KeyError(f"unknown inherited navigation task {task}")
        with torch.inference_mode():
            logits, _ = self.navigation.forward_task(task, observation)
        return int(logits.argmax(dim=-1).item())
