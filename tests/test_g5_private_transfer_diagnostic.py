import json

import pytest

from scripts.g5_private_transfer_diagnostic import LANGUAGES, OPERATIONS, load_bank


def _bank(path, *, manifest_sha="a" * 64):
    items = []
    for index, (language, operation) in enumerate(
            (language, operation) for language in LANGUAGES for operation in OPERATIONS):
        a, b = 30 + index, 7 + index
        answer = {"add": a + b, "subtract": a - b, "multiply": a * b}[operation]
        items.append({"id": f"private-g5-{language}-{operation}", "area": "arithmetic",
                      "language": language, "operation": operation, "a": a, "b": b,
                      "question": f"Compute {a} and {b} using {language} {operation}.",
                      "options": [str(answer), str(answer + 1), str(answer - 1), str(answer + 2)],
                      "answer": str(answer),
                      "key_source": "deterministic_integer_arithmetic_v1"})
    path.write_text(json.dumps({"schema": "toddler-g5-arithmetic-transfer-diagnostic/v1",
                                "status": "sealed_diagnostic_not_promotion",
                                "candidate_manifest_sha256": manifest_sha, "items": items}))
    path.chmod(0o600)
    return items


def test_private_transfer_bank_recomputes_keys_and_requires_new_forms(tmp_path):
    path = tmp_path / "bank.json"
    items = _bank(path)
    assert len(load_bank(path, manifest_sha256="a" * 64, min_per_cell=1)) == 9
    items[0]["answer"] = items[0]["options"][1]
    path.write_text(json.dumps({"schema": "toddler-g5-arithmetic-transfer-diagnostic/v1",
                                "status": "sealed_diagnostic_not_promotion",
                                "candidate_manifest_sha256": "a" * 64, "items": items}))
    with pytest.raises(ValueError, match="deterministic verification"):
        load_bank(path, manifest_sha256="a" * 64, min_per_cell=1)


def test_private_transfer_bank_rejects_wrong_candidate_and_public_form(tmp_path):
    path = tmp_path / "bank.json"
    items = _bank(path)
    with pytest.raises(ValueError, match="exact G5 candidate"):
        load_bank(path, manifest_sha256="b" * 64, min_per_cell=1)
    items[0]["question"] = "What is 30 plus 7?"
    path.write_text(json.dumps({"schema": "toddler-g5-arithmetic-transfer-diagnostic/v1",
                                "status": "sealed_diagnostic_not_promotion",
                                "candidate_manifest_sha256": "a" * 64, "items": items}))
    with pytest.raises(ValueError, match="trained exact form"):
        load_bank(path, manifest_sha256="a" * 64, min_per_cell=1)


def test_private_transfer_bank_must_be_owner_only(tmp_path):
    path = tmp_path / "bank.json"
    _bank(path)
    path.chmod(0o644)
    with pytest.raises(ValueError, match="owner-only"):
        load_bank(path, manifest_sha256="a" * 64, min_per_cell=1)
