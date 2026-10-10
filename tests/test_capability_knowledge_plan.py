"""Verify the storage assumptions and that graph links do not assert mastery."""

import hashlib
import json
from pathlib import Path

import networkx as nx
import pytest

from scripts.capability_lightrag import payload
from scripts.capability_lookup import lookup
from scripts.capability_storage_budget import estimate


ROOT = Path(__file__).resolve().parents[1]


def test_balanced_storage_estimate_counts_shared_topics_once():
    row = estimate(10_000, 100, topic_bundle_gb=8, unique_skill_mb=100, index_gb=50)
    assert row["logical_total_tb"] == 81.05
    assert row["primary_tb"] == 1.85
    assert row["physical_amortized_per_slot_mb"] == 185


def test_every_capability_has_a_discovery_link_and_no_impl_claim():
    ids = (ROOT / "knowledge" / "CAPABILITY_INDEX.txt").read_text().splitlines()
    rows = [json.loads(line) for line in
            (ROOT / "knowledge" / "graph" / "capability_repository_links.jsonl").read_text().splitlines()]
    assert [row["capability_id"] for row in rows] == ids
    assert len(rows) == 10_000
    assert all(row["repository_relation"] == "domain_discovery_candidate"
               and row["codegraph_relation"] == "related_local_context" for row in rows)
    assert sum(row["lesson_status"] == "lesson_draft" for row in rows) == 10
    connections = json.loads((ROOT / "knowledge" / "graph" / "codegraph_connections.json").read_text())
    context = json.loads((ROOT / "knowledge" / "graph" / "toddler_codegraph_context.json").read_text())
    context_nodes = {node["id"] for node in context["nodes"]}
    assert context["source_graph_sha256"] == connections["source_graph_sha256"]
    assert all(edge["from"] in context_nodes and edge["to"] in context_nodes
               for edge in context["edges"])
    for link in connections["domain_links"].values():
        assert link["local_codegraph_node"] in context_nodes
        path = ROOT / link["local_codegraph_node"].removeprefix("file:")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == link["source_sha256"]


def test_lightrag_pilot_excludes_empty_scaffolds_and_has_real_endpoints():
    data = payload(ROOT)
    names = {row["entity_name"] for row in data["entities"]}
    assert len(data["chunks"]) == 10
    assert all(row["src_id"] in names and row["tgt_id"] in names
               for row in data["relationships"])
    assert all("mastery" not in row.get("description", "").lower()
               or "unverified" in row["description"] for row in data["entities"])


def test_tsne_artifact_tracks_the_current_ten_lessons():
    artifact = json.loads((ROOT / "knowledge" / "graph" / "lesson_tsne.json").read_text())
    assert artifact["status"] == "exploratory_visualization_only"
    assert artifact["documents"] == 10
    for row in artifact["nodes"]:
        lesson = ROOT / "knowledge" / "capabilities" / row["id"] / "lesson.md"
        assert hashlib.sha256(lesson.read_bytes()).hexdigest() == row["lesson_sha256"]


def test_editorial_prerequisites_are_acyclic_and_reference_lessons():
    data = json.loads((ROOT / "knowledge" / "graph" / "prerequisites.json").read_text())
    graph = nx.DiGraph((edge["from"], edge["to"]) for edge in data["edges"])
    assert nx.is_directed_acyclic_graph(graph)
    assert all((ROOT / "knowledge" / "capabilities" / capability_id / "lesson.md").is_file()
               for capability_id in graph.nodes)


def test_lookup_distinguishes_a_lesson_from_an_empty_scaffold():
    rows = (ROOT / "knowledge" / "CAPABILITY_INDEX.txt").read_text().splitlines()
    lesson = next(row for row in rows if (ROOT / "knowledge" / "capabilities" / row / "lesson.md").is_file())
    scaffold = next(row for row in rows if not (ROOT / "knowledge" / "capabilities" / row / "lesson.md").is_file())
    assert lookup(lesson)["lesson"] is not None
    assert lookup(scaffold)["lesson"] is None
    assert lookup(scaffold)["links"]["repository_relation"] == "domain_discovery_candidate"
    with pytest.raises(KeyError):
        lookup("unregistered/capability")
