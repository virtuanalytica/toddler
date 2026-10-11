"""Cron-safe research queue for the 10,000 Toddler competency specifications.

The queue moves every ID through collection. Source candidates and lesson drafts
never become reviewed lessons or mastery evidence automatically.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    from scripts.capability_atlas import TOPICS, validate as validate_atlas
except ModuleNotFoundError:  # direct script invocation
    from capability_atlas import TOPICS, validate as validate_atlas


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = Path("/media/knight2/claude-data/knight1/knowledge/capability-atlas/pipeline")
TOPIC_IDS = {f"{domain}/{topic}" for domain, topics in TOPICS.items() for topic in topics}
SOURCE_STATUSES = {"assistant_checked_primary", "human_reviewed"}
STAGES = ("needs_sources", "needs_lesson", "needs_review", "reviewed")


def canonical_json(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


@contextmanager
def locked(state_dir: Path):
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / ".pipeline.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("knowledge pipeline is already running") from exc
        yield


def load_sources(root: Path = ROOT) -> dict[str, dict]:
    path = root / "knowledge" / "topic_sources.jsonl"
    registry: dict[str, dict] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        topic_id = row.get("topic_id")
        sources = row.get("sources")
        if (topic_id not in TOPIC_IDS or topic_id in registry
                or row.get("status") not in SOURCE_STATUSES
                or not isinstance(sources, list) or not sources):
            raise ValueError(f"invalid topic source row {number}")
        date.fromisoformat(row["checked_utc"])
        urls = set()
        for source in sources:
            url = source.get("url")
            if (not isinstance(url, str) or not url.startswith("https://")
                    or url in urls or not source.get("title") or not source.get("publisher")
                    or not source.get("role") or source.get("copy_policy") != "link_only"):
                raise ValueError(f"invalid source in row {number}")
            urls.add(url)
        registry[topic_id] = row
    return registry


def load_atlas(root: Path = ROOT) -> tuple[list[str], dict[str, dict], str]:
    validate_atlas(root / "knowledge" / "capabilities")
    index = root / "knowledge" / "CAPABILITY_INDEX.txt"
    ids = index.read_text().splitlines()
    records = {}
    for capability_id in ids:
        source = root / "knowledge" / "capabilities" / capability_id / "capability.json"
        records[capability_id] = json.loads(source.read_text())
    return ids, records, digest(index)


def stage(record: dict, sources: dict[str, dict]) -> str:
    if record["status"] == "reviewed_lesson":
        return "reviewed"
    if record["status"] == "lesson_draft":
        return "needs_review"
    if f"{record['domain']}/{record['topic']}" in sources:
        return "needs_lesson"
    return "needs_sources"


def inventory(root: Path = ROOT) -> dict:
    ids, records, index_sha = load_atlas(root)
    sources = load_sources(root)
    counts = {name: 0 for name in STAGES}
    for capability_id in ids:
        counts[stage(records[capability_id], sources)] += 1
    return {"schema": "toddler-knowledge-inventory/v1", "total": len(ids),
            "topics": len(TOPIC_IDS), "topics_with_sources": len(sources),
            "stages": counts, "index_sha256": index_sha,
            "source_register_sha256": digest(root / "knowledge" / "topic_sources.jsonl"),
            "fully_reviewed": counts["reviewed"] == len(ids)}


def read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events = []
    previous = "0" * 64
    days = set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        row = json.loads(line)
        claimed = row.pop("event_sha256", None)
        actual = hashlib.sha256(canonical_json(row)).hexdigest()
        if (claimed != actual or row.get("previous_sha256") != previous
                or row.get("day") in days or row.get("count", 0) < 1):
            raise ValueError(f"invalid knowledge ledger event {number}")
        row["event_sha256"] = claimed
        previous = claimed
        days.add(row["day"])
        events.append(row)
    return events


def tick(root: Path = ROOT, state_dir: Path = DEFAULT_STATE, *, daily_target: int = 64,
         day: date | None = None) -> dict:
    """Emit one auditable daily batch; repeat calls on the same day are inert."""
    if not 1 <= daily_target <= 1_000:
        raise ValueError("daily_target must be between 1 and 1000")
    day = day or datetime.now().astimezone().date()
    with locked(state_dir):
        ledger_path = state_dir / "ledger.jsonl"
        history = read_ledger(ledger_path)
        if history and history[-1]["day"] == day.isoformat():
            return {"status": "already_queued", **history[-1]}
        if history and history[-1]["day"] > day.isoformat():
            raise ValueError("clock moved backwards relative to the knowledge ledger")
        ids, records, index_sha = load_atlas(root)
        sources = load_sources(root)
        position = sum(event["count"] for event in history) % len(ids)
        selected = [ids[(position + offset) % len(ids)] for offset in range(daily_target)]
        rows = []
        missing_topics = set()
        counts = {name: 0 for name in STAGES}
        for capability_id in selected:
            record = records[capability_id]
            topic_id = f"{record['domain']}/{record['topic']}"
            current_stage = stage(record, sources)
            counts[current_stage] += 1
            if current_stage == "needs_sources":
                missing_topics.add(topic_id)
            rows.append({"capability_id": capability_id, "stage": current_stage,
                         "lesson_status": record["status"], "competency": record["competency"],
                         "topic_sources": sources.get(topic_id), "mastery_claim": False,
                         "instruction": ("Find and verify a primary source for the topic before drafting."
                                         if current_stage == "needs_sources" else
                                         "Draft a sourced lesson and practical exercise; keep private tests separate."
                                         if current_stage == "needs_lesson" else
                                         "Request independent source, task and safety review."
                                         if current_stage == "needs_review" else
                                         "Maintain source and assessment freshness; no automatic learner mastery.")})
        queue_path = state_dir / "queues" / f"{day.isoformat()}.jsonl"
        queue_bytes = b"".join(canonical_json(row) + b"\n" for row in rows)
        atomic_write(queue_path, queue_bytes)
        event = {"schema": "toddler-knowledge-tick/v1", "day": day.isoformat(),
                 "created_utc": datetime.now(timezone.utc).isoformat(),
                 "count": len(rows), "first_id": selected[0], "last_id": selected[-1],
                 "position_before": position, "index_sha256": index_sha,
                 "source_register_sha256": digest(root / "knowledge" / "topic_sources.jsonl"),
                 "queue": str(queue_path), "queue_sha256": hashlib.sha256(queue_bytes).hexdigest(),
                 "stages": counts, "missing_topic_sources": sorted(missing_topics),
                 "previous_sha256": history[-1]["event_sha256"] if history else "0" * 64}
        event["event_sha256"] = hashlib.sha256(canonical_json(event)).hexdigest()
        with ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        atomic_write(state_dir / "inventory.json", (json.dumps(inventory(root), indent=2) + "\n").encode())
        return {"status": "queued", **event}


def discover(root: Path = ROOT, state_dir: Path = DEFAULT_STATE, *, max_topics: int = 2) -> dict:
    """Fetch bibliographic candidates only. No source or lesson is approved here."""
    if not 1 <= max_topics <= 10:
        raise ValueError("max_topics must be 1..10")
    with locked(state_dir):
        approved = load_sources(root)
        candidate_dir = state_dir / "source_candidates"
        remaining = [topic for topic in sorted(TOPIC_IDS)
                     if topic not in approved and not (candidate_dir / f"{topic}.json").is_file()]
        written = []
        for topic_id in remaining[:max_topics]:
            query = topic_id.replace("/", " ").replace("_", " ")
            params = urlencode({"search": query, "per-page": 5,
                                "select": "id,doi,display_name,publication_year,type,primary_location"})
            request = Request("https://api.openalex.org/works?" + params,
                              headers={"User-Agent": "ToddlerKnowledgeAtlas/1.0 (research candidates; github.com/virtuanalytica/toddler)"})
            with urlopen(request, timeout=25) as response:
                items = json.load(response)["results"]
            candidates = [{"doi": item.get("doi"), "title": item.get("display_name"),
                           "venue": (item.get("primary_location") or {}).get("source", {}).get("display_name")
                           if (item.get("primary_location") or {}).get("source") else None,
                           "type": item.get("type"), "url": (item.get("primary_location") or {}).get("landing_page_url"),
                           "publication_year": item.get("publication_year"), "openalex_id": item.get("id")}
                          for item in items]
            path = candidate_dir / f"{topic_id}.json"
            atomic_write(path, (json.dumps({"topic_id": topic_id, "status": "candidate_unverified",
                                            "query": query, "provider": "OpenAlex Works API",
                                            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                                            "candidates": candidates}, ensure_ascii=False, indent=2) + "\n").encode())
            written.append(str(path))
        return {"candidate_topics_written": len(written), "candidate_files": written,
                "remaining_without_candidates": len(remaining) - len(written),
                "promotion": "none; verify relevance and primary source manually"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("audit", "tick", "discover"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--daily-target", type=int, default=64)
    parser.add_argument("--max-topics", type=int, default=2)
    args = parser.parse_args()
    if args.action == "audit":
        result = inventory(args.root)
        history = read_ledger(args.state_dir / "ledger.jsonl")
        result["queued_visits"] = sum(event["count"] for event in history)
        result["last_tick"] = history[-1]["day"] if history else None
    elif args.action == "tick":
        result = tick(args.root, args.state_dir, daily_target=args.daily_target)
    else:
        result = discover(args.root, args.state_dir, max_topics=args.max_topics)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
