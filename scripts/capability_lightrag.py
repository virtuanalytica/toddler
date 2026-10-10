"""Build a sourced LightRAG custom-KG pilot from the ten lesson drafts."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT = Path("/media/knight2/claude-data/knight1/knowledge/capability-atlas/lightrag-pilot")


def payload(root: Path = ROOT) -> dict:
    links = root / "knowledge" / "graph" / "capability_repository_links.jsonl"
    rows = [json.loads(line) for line in links.read_text().splitlines()]
    selected = [row for row in rows if row["lesson_status"] == "lesson_draft"]
    if not selected:
        raise ValueError("no sourced lessons available for LightRAG")
    chunks, entities, relationships = [], {}, []
    for row in selected:
        capability_id = row["capability_id"]
        path = root / "knowledge" / "capabilities" / capability_id / "lesson.md"
        alias = "lesson:" + capability_id
        chunks.append({"content": path.read_text(), "source_id": alias,
                       "file_path": path.relative_to(root).as_posix()})
        cap, repo, code = "CAP:" + capability_id, "REPO:" + row["repository"], "CODE:" + row["codegraph_node"]
        entities[cap] = {"entity_name": cap, "entity_type": "capability_lesson",
                         "description": capability_id + " lesson draft; mastery unverified",
                         "source_id": alias, "file_path": path.relative_to(root).as_posix()}
        for name, kind, description in ((repo, "repository_candidate", row["repository"]),
                                        (code, "local_codegraph_node", row["codegraph_node"])):
            entities.setdefault(name, {"entity_name": name, "entity_type": kind,
                                       "description": description, "source_id": alias})
        relationships.extend((
            {"src_id": cap, "tgt_id": repo, "description": "domain-level repository discovery candidate",
             "keywords": "repository candidate", "source_id": alias},
            {"src_id": cap, "tgt_id": code, "description": "verified local codegraph context, not implementation proof",
             "keywords": "codegraph context", "source_id": alias}))
        for url in row["lesson_sources"]:
            source = "SOURCE:" + url
            entities.setdefault(source, {"entity_name": source, "entity_type": "primary_source",
                                        "description": url, "source_id": alias})
            relationships.append({"src_id": cap, "tgt_id": source,
                                  "description": "cited in the draft lesson", "keywords": "source citation",
                                  "source_id": alias})
    return {"chunks": chunks, "entities": list(entities.values()), "relationships": relationships}


async def ingest(pilot: Path = PILOT, root: Path = ROOT) -> dict:
    """Disposable local pilot; custom-KG writes are rebuilt from this manifest after a crash."""
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    from sentence_transformers import SentenceTransformer

    if pilot.exists() and any(pilot.iterdir()):
        raise FileExistsError(f"pilot directory is not empty: {pilot}")
    pilot.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)

    async def embed(texts: list[str]):
        return await asyncio.to_thread(lambda: model.encode(texts, normalize_embeddings=True))

    async def no_llm(*_args, **_kwargs):
        raise RuntimeError("pilot indexes curated graph records; generative querying is not configured")

    rag = LightRAG(working_dir=str(pilot), embedding_func=EmbeddingFunc(384, embed,
                   max_token_size=512, model_name="sentence-transformers/all-MiniLM-L6-v2"),
                   llm_model_func=no_llm)
    await rag.initialize_storages()
    try:
        data = payload(root)
        await rag.ainsert_custom_kg(data)
        return {"status": "indexed_custom_kg_pilot", "chunks": len(data["chunks"]),
                "entities": len(data["entities"]), "relationships": len(data["relationships"]),
                "working_dir": str(pilot)}
    finally:
        await rag.finalize_storages()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("export", "ingest"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pilot", type=Path, default=PILOT)
    args = parser.parse_args()
    result = payload() if args.action == "export" else asyncio.run(ingest(args.pilot))
    body = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body)
    else:
        print(body, end="")
