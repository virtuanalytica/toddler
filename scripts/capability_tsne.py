"""Map sourced lesson drafts for exploration; similarity is not mastery evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]


def build(root: Path = ROOT) -> dict:
    lessons = sorted((root / "knowledge" / "capabilities").rglob("lesson.md"))
    if len(lessons) < 6:
        raise ValueError("at least six reviewed-source lesson drafts are needed for this pilot")
    documents = [p.read_text(encoding="utf-8") for p in lessons]
    vectors = TfidfVectorizer(ngram_range=(1, 2), max_features=4000, sublinear_tf=True).fit_transform(documents)
    similarity = cosine_similarity(vectors)
    distances = np.clip(1 - similarity, 0, 2)
    np.fill_diagonal(distances, 0)
    perplexity = min(5, (len(lessons) - 1) / 3)
    coords = TSNE(n_components=2, metric="precomputed", init="random", perplexity=perplexity,
                  random_state=42, max_iter=1000).fit_transform(distances)
    coords = (coords - coords.min(axis=0)) / np.maximum(np.ptp(coords, axis=0), 1e-12)
    rows = []
    for i, path in enumerate(lessons):
        capability_id = path.parent.relative_to(root / "knowledge" / "capabilities").as_posix()
        neighbors = sorted((j for j in range(len(lessons)) if j != i),
                           key=lambda j: (-similarity[i, j], j))[:3]
        rows.append({"id": capability_id, "lesson_sha256": hashlib.sha256(documents[i].encode()).hexdigest(),
                     "x": round(float(coords[i, 0]), 6), "y": round(float(coords[i, 1]), 6),
                     "nearest_by_source_text": [{"id": lessons[j].parent.relative_to(root / "knowledge" / "capabilities").as_posix(),
                                                 "cosine": round(float(similarity[i, j]), 6)} for j in neighbors]})
    return {"schema": "toddler-capability-tsne-pilot/v1", "status": "exploratory_visualization_only",
            "documents": len(lessons), "representation": "TF-IDF uni+bi-grams of sourced lesson drafts",
            "sklearn_version": sklearn.__version__, "random_state": 42,
            "perplexity": perplexity,
            "trustworthiness_k3": round(float(trustworthiness(vectors.toarray(), coords,
                                                              n_neighbors=3, metric="cosine")), 6),
            "nodes": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build(args.root)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    else:
        print(payload, end="")
