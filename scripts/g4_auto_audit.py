"""Resume a single prospective G4-search audit after Teacher finishes.

The public gate runs before private seeds are made. A family reservation
permits one audited candidate only; interrupted trials resume the same
protocol, weights and committed seed sets. No hidden seed is printed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from scripts import g4_search_private_trial as G4

DEFAULT_TEACHER_RUNS = Path("~/.local/share/teacher/g4-search").expanduser()
DEFAULT_AUDITS = Path("~/.local/share/toddler/g4-search-private").expanduser()


def latest_completed_manifest(runs: Path, *, max_age_hours: float = 6) -> Path | None:
    """Use the run that just finished, never an older attractive candidate."""
    directories = sorted(runs.glob("*/"), reverse=True)
    if not directories:
        return None
    path = directories[0] / "manifest.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "completed_candidate":
        return None
    stamp = manifest.get("created_utc", "").split("-", 1)[0]
    try:
        created = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    age_hours = (datetime.now(timezone.utc) - created).total_seconds() / 3600
    return path if 0 <= age_hours <= max_age_hours else None


def audit(manifest_path: Path, family_root: Path, registry_root: Path) -> dict:
    """Prepare, run or resume, assess, and import only a replicated winner."""
    manifest, _ = G4.validate_cohort(manifest_path, registry_root)
    cohort_id = manifest["created_utc"]
    gate = G4.public_gate(manifest_path, manifest)
    if not gate["eligible"]:
        return {"status": "public_gate_failed", "cohort": cohort_id,
                "reason": gate["reason"], "selected_children": gate.get("selected_children"),
                "private_seeds_created": False}

    family_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = family_root / "family-attempt.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        if lock.get("manifest_sha256") != G4.digest(manifest_path):
            return {"status": "family_attempt_already_spent", "cohort": cohort_id,
                    "private_seeds_created_for_this_cohort": False}
        protocol = Path(lock["out"]) / "protocol.json"
        if not protocol.is_file():
            raise RuntimeError("family reservation exists without protocol; recover reservation before retry")
    else:
        out = family_root / ("audit-" + cohort_id)
        os.environ["TODDLER_G4_SEARCH_TRIAL_ROOT"] = str(family_root.resolve())
        prepared = G4.prepare(manifest_path, out, registry_root)
        protocol = Path(prepared["protocol"])

    for index in (0, 1):
        if not (protocol.parent / f"trial-{index + 1}.json").is_file():
            G4.run(protocol, index, registry_root)
    decision = G4.assess(protocol, registry_root)
    if not decision["promote"]:
        return {"status": "private_gate_failed", "cohort": cohort_id,
                "protocol_sha256": G4.digest(protocol), "alpha": decision["alpha"],
                "trial_passed": [row["passed"] for row in decision["trials"]],
                "private_seeds_created": True}
    imported = G4.promote(protocol, registry_root)
    return {"status": "promoted", "cohort": cohort_id,
            "protocol_sha256": G4.digest(protocol), "generation": imported["generation"],
            "children": imported["children"], "ledger_verified": imported["ledger_verified"]}


def write_status(family_root: Path, cohort: str, result: dict) -> Path:
    family_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    status_dir = family_root / "status"
    status_dir.mkdir(mode=0o700, exist_ok=True)
    path = status_dir / f"{cohort}.json"
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex[:8])
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--runs", type=Path, default=DEFAULT_TEACHER_RUNS)
    parser.add_argument("--audits", type=Path, default=DEFAULT_AUDITS)
    parser.add_argument("--registry", type=Path, default=G4.ROOT)
    args = parser.parse_args()
    manifest = args.manifest or latest_completed_manifest(args.runs)
    if manifest is None:
        print(json.dumps({"status": "no_recent_completed_cohort",
                          "recovery": "inspect the latest Teacher run and rerun with --manifest after repairing it"}))
        return 1
    cohort = manifest.parent.name
    try:
        result = audit(manifest, args.audits, args.registry)
    except Exception as exc:
        result = {"status": "failed", "cohort": cohort,
                  "reason": f"{type(exc).__name__}: {exc}",
                  "recovery": "inspect the reserved protocol and any incomplete trial file, then rerun with --manifest; never create replacement seeds for the same attempt"}
    status_path = write_status(args.audits, cohort, result)
    print(json.dumps({"status": result["status"], "cohort": cohort,
                      "status_path": str(status_path)}, ensure_ascii=False))
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
