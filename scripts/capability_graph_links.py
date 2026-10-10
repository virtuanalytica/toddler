"""Connect every capability address to source repositories and a verified local codegraph node."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH = Path("/media/knight2/claude-data/knight1/knowledge/capability-atlas/graphs/"
                     "toddler-wt-g3-survived/code_graph/toddler_wt_g3_survived_gitnexus.json")

# Domain links are discovery candidates, never proof that each specific slot is implemented.
DOMAIN_LINKS = {
    "neuroscience": ("mne-tools/mne-python", "docs/design/brain.md"),
    "learning_theory": ("scikit-learn/scikit-learn", "toddler/learn/multitask.py"),
    "neural_networks": ("pytorch/pytorch", "toddler/learn/multitask.py"),
    "reinforcement_learning": ("Farama-Foundation/Minigrid", "toddler/learn/ppo.py"),
    "continual_learning": ("ContinualAI/avalanche", "toddler/learn/multitask.py"),
    "memory_systems": ("HKUDS/LightRAG", "toddler/knowledge/weave.py"),
    "reasoning_and_language": ("huggingface/transformers", "toddler/g4_curriculum.py"),
    "multimodal_agents": ("Farama-Foundation/Minigrid", "toddler/learn/tasks.py"),
    "evaluation_and_causality": ("google-research/rliable", "toddler/learn/ancestor_gate.py"),
    "systems_and_safety": ("virtuanalytica/virtualv_llm", "toddler/stop.py"),
}


def build(root: Path = ROOT, graph_path: Path = DEFAULT_GRAPH) -> dict:
    graph_bytes = graph_path.read_bytes()
    graph = json.loads(graph_bytes)
    if not graph.get("nodes") or not graph.get("edges"):
        raise ValueError("codegraph is empty")
    files = {node["id"]: node for node in graph["nodes"] if node.get("type") == "source_file"}
    domains = {}
    for domain, (repo, path) in DOMAIN_LINKS.items():
        node_id = f"file:{path}"
        node = files.get(node_id)
        if node is None or node.get("path") != path:
            raise ValueError(f"{domain} has no verified local source-file node")
        current_sha = hashlib.sha256((root / path).read_bytes()).hexdigest()
        if current_sha != node["sha256"]:
            raise ValueError(f"stale codegraph for {path}")
        domains[domain] = {"repository": f"https://github.com/{repo}",
                           "repository_relation": "domain_discovery_candidate",
                           "local_codegraph_node": node_id,
                           "codegraph_relation": "related_local_context",
                           "source_sha256": current_sha}
    ids = (root / "knowledge" / "CAPABILITY_INDEX.txt").read_text().splitlines()
    if len(ids) != 10_000 or len(set(ids)) != 10_000:
        raise ValueError("canonical atlas index needs 10,000 unique entries")
    out = root / "knowledge" / "graph"
    out.mkdir(exist_ok=True)
    graph_sha = hashlib.sha256(graph_bytes).hexdigest()
    mapping = {"schema": "toddler-capability-codegraph-links/v1",
               "source_graph_sha256": graph_sha, "source_graph_schema": graph.get("schema"),
               "source_graph_nodes": len(graph["nodes"]), "source_graph_edges": len(graph["edges"]),
               "domain_links": domains}
    (out / "codegraph_connections.json").write_text(json.dumps(mapping, indent=2) + "\n")
    # Keep the traversable one-hop evidence in Git. The full graph stays on
    # claude-data; this subset is enough to inspect each linked file's symbols,
    # imports and direct calls without copying the complete external index.
    seed_files = {link["local_codegraph_node"] for link in domains.values()}
    selected = set(seed_files)
    context_edges = []
    for edge in graph["edges"]:
        if edge["from"] in seed_files and edge["kind"] in ("defines", "imports"):
            selected.add(edge["to"])
            context_edges.append(edge)
    defined_symbols = {edge["to"] for edge in context_edges if edge["kind"] == "defines"}
    for edge in graph["edges"]:
        if edge["from"] in defined_symbols and edge["kind"] == "calls":
            selected.add(edge["to"])
            context_edges.append(edge)
    context = {"schema": "toddler-capability-codegraph-context/v1", "source_graph_sha256": graph_sha,
               "scope": "domain-linked files, their symbols/imports and one direct call hop",
               "nodes": [node for node in graph["nodes"] if node["id"] in selected],
               "edges": context_edges}
    (out / "toddler_codegraph_context.json").write_text(json.dumps(context, ensure_ascii=False) + "\n")
    with (out / "capability_repository_links.jsonl").open("w") as handle:
        for capability_id in ids:
            domain, topic, _ = capability_id.split("/", 2)
            link = domains[domain]
            lesson = root / "knowledge" / "capabilities" / capability_id / "lesson.md"
            metadata = json.loads((lesson.parent / "capability.json").read_text()) if lesson.is_file() else None
            row = {"capability_id": capability_id, "topic_id": f"{domain}/{topic}",
                   "repository": link["repository"], "repository_relation": link["repository_relation"],
                   "codegraph_node": link["local_codegraph_node"],
                   "codegraph_relation": link["codegraph_relation"],
                   "lesson_sources": metadata["evidence"]["primary_sources"] if metadata else [],
                   "lesson_status": metadata["status"] if metadata else "scaffold"}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"linked_capabilities": len(ids), "topics": len({"/".join(x.split("/")[:2]) for x in ids}),
            "domain_repositories": len(set(x["repository"] for x in domains.values())),
            "graph_sha256": graph_sha,
            "verified_lesson_links": sum((root / "knowledge" / "capabilities" / x / "lesson.md").is_file()
                                         for x in ids)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.graph)))
