"""Knowledge-base tests on the committed corpus (real Wikipedia extracts with hashes) and a code
graph listing this repository's real tracked files."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from toddler.knowledge import concepts, weave

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "data" / "corpus" / "concepts"


def _code_graph(root: Path) -> None:
    files = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
    graph = {"nodes": [{"id": f"file:{f}", "type": "source_file", "path": f, "summary": f} for f in files],
             "edges": []}
    (root / "code_graph").mkdir(parents=True)
    (root / "code_graph" / "toddler_gitnexus.json").write_text(json.dumps(graph))


def test_curated_structure_is_consistent():
    for a, b, _ in concepts.RELATIONS:
        assert a in concepts.CONCEPTS and b in concepts.CONCEPTS
    for cid, (target, _why) in concepts.TODDLER_LINKS.items():
        assert cid in concepts.CONCEPTS
        assert (REPO / target.split("#")[0]).exists(), target


def test_committed_corpus_hashes_match():
    for cid in concepts.CONCEPTS:
        doc = json.loads((CORPUS / f"{cid}.json").read_text(encoding="utf-8"))
        assert weave.sha(doc["extract"]) == doc["sha256"] and doc["licence"]


def test_weave_links_all_25_rows_without_problems(tmp_path, monkeypatch):
    monkeypatch.setattr(weave, "BUILDER", "")
    _code_graph(tmp_path)
    g = weave.weave(CORPUS, tmp_path)
    assert g["metadata"]["problems"] == [] and g["metadata"]["rows"] == 25
    rows_linked = {e["from"] for e in g["edges"] if e["kind"] == "implemented_in"}
    assert rows_linked == {f"row:{n}" for n in range(1, 26)}


def test_weave_reports_edited_corpus_file(tmp_path, monkeypatch):
    monkeypatch.setattr(weave, "BUILDER", "")
    corpus = tmp_path / "corpus"
    shutil.copytree(CORPUS, corpus)
    doc = json.loads((corpus / "perceptron.json").read_text())
    doc["extract"] += " edited"
    (corpus / "perceptron.json").write_text(json.dumps(doc))
    _code_graph(tmp_path / "out")
    g = weave.weave(corpus, tmp_path / "out")
    assert any("perceptron" in p and "sha256" in p for p in g["metadata"]["problems"])


def test_weave_needs_code_graph(tmp_path):
    with pytest.raises(FileNotFoundError, match="code graph"):
        weave.weave(CORPUS, tmp_path)


def test_bundle_round_trip_and_unsigned_marker(tmp_path, monkeypatch):
    publish = pytest.importorskip("toddler.knowledge.publish", reason="knitweb not available")
    monkeypatch.setattr(weave, "BUILDER", "")
    _code_graph(tmp_path)
    g = weave.weave(CORPUS, tmp_path)
    woven = tmp_path / "woven" / "toddler_woven_gitnexus.json"
    data, digest = publish.build_bundle(woven, "test")
    shared = [e for e in g["edges"] if e["kind"] not in publish.PRIVATE_KINDS]
    assert len(publish.relations_from_graph(g)) == len(shared) and len(digest) >= 32
    assert publish.write_bundle(tmp_path / "b", data).name.endswith(".synaptic.unsigned")
    signed = publish.write_bundle(tmp_path / "b", data, sign=lambda d: "sig")
    assert signed.suffix == ".synaptic" and signed.with_suffix(".synaptic.sig").read_text() == "sig"
