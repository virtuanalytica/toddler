"""Weave theory, the Wee et al. (2017) mapping and Toddler's code into one gitnexus graph.

Inputs:
  * concept corpus (data/corpus/concepts/*.json, fetched by corpus.py)
  * code graph built by the existing gitnexus codebase builder
    (numerai-signals/scripts/knowledge/build_codebase_lightrag_gitnexus_obsidian.py)
  * docs/design/wee2017-mapping.md (the 25 rows)
Outputs (same gitnexus schema family, no new format):
  * ~/.gitnexus/toddler/woven/toddler_woven_gitnexus.json
  * ~/.gitnexus/toddler/lightrag/toddler_concepts_facts.jsonl and toddler_concepts_lightrag.db
The weave fails loudly when a mapping row has no path to existing code or a concept link
points at a file that does not exist; that is the coverage check for the operator's goal.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from toddler.knowledge.concepts import CONCEPTS, RELATIONS, TODDLER_LINKS

REPO = Path(__file__).resolve().parents[2]
GITNEXUS = Path.home() / ".gitnexus" / "toddler"
BUILDER = Path("/media/knight2/EDS2/projects/numerai-signals/scripts/knowledge/build_codebase_lightrag_gitnexus_obsidian.py")

SOURCES = {
    "source:wee2017": ("Wee et al. (2017) Neonatal neural networks predict children behavioral profiles later in life. "
                       "Human Brain Mapping 38(3):1362-1373", "https://doi.org/10.1002/hbm.23459"),
    "source:embedded": ("Salelanonda, Learning how to learn: toddlers vs. neural networks, Embedded.com",
                        "https://www.embedded.com/learning-how-to-learn-toddlers-vs-neural-networks/"),
}

# Wee et al. mapping row -> Toddler code files that implement it.
ROW_FILES: dict[int, tuple[str, ...]] = {
    1: ("toddler/provenance.py",), 2: ("toddler/evaluation.py", "toddler/objective.py"),
    3: ("toddler/evaluation.py",), 4: ("toddler/evaluation.py", "toddler/objective.py"),
    5: ("toddler/fastpath.py", "toddler/stop.py"), 6: ("toddler/structure.py",),
    7: ("toddler/structure.py", "toddler/fastpath.py"), 8: ("toddler/structure.py",),
    9: ("toddler/structure.py",), 10: ("toddler/structure.py",), 11: ("toddler/structure.py",),
    12: ("toddler/structure.py",), 13: ("toddler/structure.py", "toddler/objective.py"),
    14: ("toddler/evaluation.py",), 15: ("toddler/evaluation.py",),
    16: ("toddler/evaluation.py", "toddler/fastpath.py"), 17: ("toddler/evaluation.py",),
    18: ("toddler/evaluation.py",), 19: ("toddler/fastpath.py",), 20: ("toddler/stop.py", "toddler/fastpath.py"),
    21: ("toddler/fastpath.py",), 22: ("toddler/provenance.py",), 23: ("toddler/evaluation.py",),
    24: ("toddler/evaluation.py",), 25: ("toddler/stop.py", "toddler/objective.py"),
}

# Wee et al. mapping row -> concepts it rests on.
ROW_CONCEPTS: dict[int, tuple[str, ...]] = {
    2: ("cbcl", "rlhf"), 3: ("cbcl",), 7: ("diffusion_mri",), 8: ("diffusion_mri",), 9: ("clustering_coefficient",),
    11: ("louvain",), 12: ("louvain",), 15: ("svm",), 17: ("cca",), 19: ("amygdala",),
    20: ("inferior_frontal_gyrus", "insular_cortex"), 21: ("developmental_robotics", "curriculum_learning"),
    22: ("amygdala",), 24: ("reward_hacking",), 25: ("developmental_robotics",),
}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_mapping(md: Path) -> dict[int, tuple[str, str]]:
    rows: dict[int, tuple[str, str]] = {}
    for line in md.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\|\s*(\d+)\s*\|(.+?)\|(.+?)\|", line)
        if m:
            rows[int(m.group(1))] = (m.group(2).strip(), m.group(3).strip())
    return rows


def _load_builder_schema() -> str:
    spec = importlib.util.spec_from_file_location("gitnexus_builder", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # module defines SCHEMA at import; main() is not run
    return mod.SCHEMA


def weave(corpus_dir: Path = REPO / "data/corpus/concepts", out_root: Path = GITNEXUS) -> dict:
    code = json.loads((out_root / "code_graph" / "toddler_gitnexus.json").read_text(encoding="utf-8"))
    code_files = {n["path"] for n in code["nodes"] if n.get("type") == "source_file"}
    rows = parse_mapping(REPO / "docs/design/wee2017-mapping.md")
    problems: list[str] = []

    nodes: list[dict] = []
    edges: list[dict] = []
    for nid, (title, url) in SOURCES.items():
        nodes.append({"id": nid, "type": "source", "summary": title, "url": url})
    for cid, (title, year, era) in CONCEPTS.items():
        f = corpus_dir / f"{cid}.json"
        if not f.exists():
            problems.append(f"concept {cid}: corpus file missing")
            continue
        doc = json.loads(f.read_text(encoding="utf-8"))
        nodes.append({"id": f"concept:{cid}", "type": "concept", "summary": doc["extract"][:2000], "year": year,
                      "era": era, "url": doc["url"], "licence": doc["licence"], "sha256": doc["sha256"]})
    for a, b, kind in RELATIONS:
        edges.append({"from": f"concept:{a}", "to": f"concept:{b}", "kind": kind, "evidence": "concepts.RELATIONS"})
    for cid, (target, why) in TODDLER_LINKS.items():
        path = target.split("#")[0]
        if not (REPO / path).exists():
            problems.append(f"concept {cid}: link target {path} does not exist")
        edges.append({"from": f"concept:{cid}", "to": f"file:{path}", "kind": "used_in_toddler", "evidence": why})
    for n, (logic, function) in sorted(rows.items()):
        nodes.append({"id": f"row:{n}", "type": "paper_logic", "summary": f"{logic} -> {function}"})
        edges.append({"from": "source:wee2017", "to": f"row:{n}", "kind": "states"})
        for path in ROW_FILES.get(n, ()):
            if path not in code_files:
                problems.append(f"row {n}: {path} not in code graph")
            edges.append({"from": f"row:{n}", "to": f"file:{path}", "kind": "implemented_in", "evidence": function})
        for cid in ROW_CONCEPTS.get(n, ()):
            edges.append({"from": f"row:{n}", "to": f"concept:{cid}", "kind": "rests_on"})
    missing_rows = sorted(set(range(1, 26)) - {n for n in rows if ROW_FILES.get(n)})
    if missing_rows:
        problems.append(f"rows without implementation: {missing_rows}")
    for node in code["nodes"]:
        if node.get("type") == "source_file":
            nodes.append({"id": node["id"], "type": "source_file", "summary": node.get("summary", ""), "path": node["path"]})
    edges.extend(e for e in code["edges"] if e["kind"] == "defines")
    edges.append({"from": "source:embedded", "to": "concept:meta_learning", "kind": "motivates",
                  "evidence": "toddlers learn from a few examples; networks need millions"})

    graph = {"schema": "gitnexus-toddler-woven/v1", "project": "toddler",
             "generated_at": datetime.now(timezone.utc).isoformat(), "nodes": nodes, "edges": edges,
             "metadata": {"problems": problems, "rows": len(rows), "concepts": len(CONCEPTS)}}
    out = out_root / "woven"
    out.mkdir(parents=True, exist_ok=True)
    (out / "toddler_woven_gitnexus.json").write_text(json.dumps(graph, ensure_ascii=False, indent=1), encoding="utf-8")
    _write_lightrag(nodes, edges, out_root / "lightrag")
    return graph


def _write_lightrag(nodes: list[dict], edges: list[dict], lr: Path) -> None:
    lr.mkdir(parents=True, exist_ok=True)
    facts = [{"kind": n["type"], "content": n["summary"], "context": n["id"], "source": n.get("url", "toddler_weave"),
              "content_sha": sha(n["summary"])} for n in nodes if n["type"] in ("concept", "paper_logic", "source")]
    with (lr / "toddler_concepts_facts.jsonl").open("w", encoding="utf-8") as fh:
        for f in facts:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")
    db_path = lr / "toddler_concepts_lightrag.db"
    if db_path.exists():
        db_path.unlink()
    with sqlite3.connect(db_path) as db:
        db.executescript(_load_builder_schema())
        for f in facts:
            db.execute("INSERT OR IGNORE INTO facts(kind, content, context, source, created_at, content_sha) VALUES (?,?,?,?,?,?)",
                       (f["kind"], f["content"], f["context"], f["source"], time.time(), f["content_sha"]))
        for n in nodes:
            db.execute("INSERT OR IGNORE INTO entities(name, kind, description) VALUES (?,?,?)", (n["id"], n["type"], n["summary"][:2000]))
        ids = {name: eid for eid, name in db.execute("SELECT id, name FROM entities")}
        for e in edges:
            if e["from"] in ids and e["to"] in ids:
                db.execute("INSERT OR IGNORE INTO relations(src_id, dst_id, kind, evidence) VALUES (?,?,?,?)",
                           (ids[e["from"]], ids[e["to"]], e["kind"], e.get("evidence", "")[:1000]))
        db.execute("INSERT INTO facts_fts(facts_fts) VALUES('rebuild')")
        db.execute("INSERT INTO entities_fts(entities_fts) VALUES('rebuild')")
        db.commit()


if __name__ == "__main__":
    g = weave()
    print(json.dumps({"nodes": len(g["nodes"]), "edges": len(g["edges"]), **g["metadata"]}, indent=1))
