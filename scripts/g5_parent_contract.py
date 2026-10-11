"""Freeze and verify the five promoted G4 navigation parents for G5 development.

The output contains only public lineage and weight commitments. It contains no
private evaluation seeds, prompts or per-item scores. A later G5 trainer must
validate this contract again immediately before loading a parent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from toddler.learn.generations import Registry
from toddler.learn.lineage import Ledger
from toddler.learn.routing import TaskExpertRouter

DEFAULT_ROOT = Path("/media/knight2/EDS2/toddler-generations")
GENERATION = "G4-search"
CHILD_IDS = tuple(f"t600{i}" for i in range(1, 6))
NAV_TASKS = frozenset(("cartpole", "acrobot", "empty5", "doorkey5", "doorkey8",
                       "unlock", "unlockpickup", "keycorridor3", "lavacross9"))


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def module_digest(module: torch.nn.Module) -> str:
    """Hash tensor content, names, dtypes and shapes independently of torch.save."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(_canonical({"name": name, "dtype": str(value.dtype), "shape": list(value.shape)}))
        digest.update(b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def current_contract(root: Path) -> dict:
    root = Path(root)
    ledger = Ledger(root)
    valid, bad_seq = ledger.verify()
    if not valid:
        raise ValueError(f"lineage hash chain fails at sequence {bad_seq}")
    verdict = ledger.verdict(GENERATION)
    if not verdict or verdict["verdict"] != "survived" or verdict.get("evidence", {}).get("promote") is not True:
        raise ValueError("G4-search lacks a verified survived/promote verdict")
    evidence = verdict["evidence"]
    if len(evidence.get("trial_sha256", [])) != 2:
        raise ValueError("G4-search must have two confirmatory trial commitments")
    births = ledger.members(GENERATION)
    if {item["ref"] for item in births} != {f"{GENERATION}/{child}" for child in CHILD_IDS}:
        raise ValueError("G4-search ledger does not contain exactly the five expected parents")

    registry = Registry(root)
    rows = []
    for child in CHILD_IDS:
        ref = f"{GENERATION}/{child}"
        birth = ledger.birth(ref)
        net, record = registry.load(GENERATION, child)
        if not isinstance(net, TaskExpertRouter):
            raise ValueError(f"{ref} is not a navigation task router")
        if record.weights_sha256 != birth["weights_sha256"]:
            raise ValueError(f"{ref} registry weights differ from lineage birth")
        if net.route != record.config.get("route"):
            raise ValueError(f"{ref} route differs from registry metadata")
        if set(net.route) != NAV_TASKS:
            raise ValueError(f"{ref} does not preserve all nine navigation tasks")
        if not any(inherited["parent"] == f"G3-recombined/t500{child[-1]}"
                   for inherited in birth["inherits"]):
            raise ValueError(f"{ref} lacks the matched G3 navigation ancestor")
        for inherited in birth["inherits"]:
            if ledger.weights_of(inherited["parent"]) != inherited["parent_weights_sha256"]:
                raise ValueError(f"{ref} ancestor weight commitment differs")
        rows.append({
            "ref": ref,
            "weights_sha256": record.weights_sha256,
            "parent_refs": list(record.parents),
            "route": dict(sorted(net.route.items())),
            "route_sha256": _sha256(_canonical(net.route)),
            "expert_sha256": {name: module_digest(expert)
                              for name, expert in sorted(net.experts.items())},
        })

    return {
        "schema": "toddler-g5-frozen-g4-parents/v1",
        "generation": GENERATION,
        "selection_seq": verdict["_seq"],
        "promotion_protocol_sha256": evidence["protocol_sha256"],
        "confirmatory_trial_sha256": list(evidence["trial_sha256"]),
        "parents": rows,
    }


def verify_contract(root: Path, contract: dict) -> None:
    actual = current_contract(root)
    if contract != actual:
        raise ValueError("G5 parent contract differs from the current promoted G4 lineage or weights")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--write", type=Path, help="write a frozen public contract; refuse overwrite")
    parser.add_argument("--verify", type=Path, help="verify an existing frozen contract")
    args = parser.parse_args()
    if bool(args.write) == bool(args.verify):
        parser.error("choose exactly one of --write or --verify")
    if args.verify:
        verify_contract(args.root, json.loads(args.verify.read_text()))
        print("G5 parent contract verified: five frozen G4 navigation parents")
    else:
        contract = current_contract(args.root)
        args.write.parent.mkdir(parents=True, exist_ok=True)
        with args.write.open("x") as handle:
            json.dump(contract, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        print(f"G5 parent contract written: {args.write}")


if __name__ == "__main__":
    main()
