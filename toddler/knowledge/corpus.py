"""Fetch the concept corpus from real sources (Wikipedia REST summaries) with provenance.

Each article is stored as JSON with url, licence, retrieval date and sha256, and registered
in toddler.provenance. A concept whose article cannot be fetched is reported, never filled
with generated text.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import date
from pathlib import Path

import requests

from toddler import provenance
from toddler.knowledge.concepts import CONCEPTS

API = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
LICENCE = "CC BY-SA 4.0"
HEADERS = {"User-Agent": "toddler-knowledge/0.1 (https://github.com/virtuanalytica/toddler)"}


def fetch_all(out_dir: Path, pause_s: float = 0.2) -> tuple[provenance.Register, dict[str, str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    reg = provenance.Register()
    failures: dict[str, str] = {}
    for cid, (title, _year, _era) in CONCEPTS.items():
        path = out_dir / f"{cid}.json"
        if path.exists():
            doc = json.loads(path.read_text(encoding="utf-8"))
        else:
            r = requests.get(API.format(title=title), headers=HEADERS, timeout=20)
            if r.status_code != 200:
                failures[cid] = f"HTTP {r.status_code}"
                continue
            data = r.json()
            extract = data.get("extract", "").strip()
            if not extract:
                failures[cid] = "empty extract"
                continue
            doc = {
                "concept": cid,
                "title": data.get("title", title),
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{title}"),
                "revision": data.get("revision", ""),
                "extract": extract,
                "licence": LICENCE,
                "retrieved": date.today().isoformat(),
                "sha256": hashlib.sha256(extract.encode("utf-8")).hexdigest(),
            }
            path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
            time.sleep(pause_s)
        reg.admit(provenance.Source(cid, doc["url"], doc["licence"], date.fromisoformat(doc["retrieved"]),
                                    doc["sha256"], inclusion_rule="curated AI/NN concept list"))
    return reg, failures
