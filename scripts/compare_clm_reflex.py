"""Local CLM/JEV shadow run over private JSONL scenes; never controls hardware.

Each input row has ``state`` and ``sensors`` with force_n, speed_m_s,
person_distance_m. The output stores decisions and latencies, never raw scenes.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toddler.clm import DEFAULT_ENDPOINT, HttpClmClient
from toddler.fastpath import Sensors
from toddler.jev import HttpJevClient
from toddler.reflex_shadow import compare


def clm_ready(endpoint: str) -> dict:
    health_url = endpoint.rsplit("/v1/systemone", 1)[0] + "/health"
    response = requests.get(health_url, timeout=3)
    response.raise_for_status()
    body = response.json()
    if body.get("mock") or body.get("ok") is not True or body.get("embedder") is not True:
        raise RuntimeError(f"CLM backend not ready or mock: {body!r}")
    return body


def _percentile(values: list[float], percent: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, (percent * len(ordered) + 99) // 100 - 1)], 3)


def run(input_path: Path, output_path: Path, jev: HttpJevClient, clm: HttpClmClient) -> dict:
    records = []
    for index, line in enumerate(input_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        scene = json.loads(line)
        sensors = Sensors(**scene["sensors"])
        result = compare(scene["state"], sensors, jev, clm)
        records.append({"index": index, "jev": result.jev.command, "clm_shadow": result.clm.command,
                        "jev_ms": round(result.jev_ms, 3), "clm_ms": round(result.clm_ms, 3),
                        "jev_error": result.jev_error, "clm_error": result.clm_error})
    output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    paired = [r for r in records if not r["jev_error"] and not r["clm_error"]]
    latencies = [r["clm_ms"] for r in records if not r["clm_error"]]
    return {"items": len(records), "paired": len(paired),
            "disagreements": sum(r["jev"] != r["clm_shadow"] for r in paired),
            "jev_errors": sum(bool(r["jev_error"]) for r in records),
            "clm_errors": sum(bool(r["clm_error"]) for r in records),
            "clm_p50_ms": round(statistics.median(latencies), 3) if latencies else None,
            "clm_p95_ms": _percentile(latencies, 95), "clm_p99_ms": _percentile(latencies, 99),
            "clm_over_20ms": sum(r["clm_ms"] > 20 for r in records),
            "status": "paired" if len(paired) == len(records) else "incomplete"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--clm-endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--jev-endpoint", default="http://127.0.0.1:8092/v1/systemone")
    args = parser.parse_args()
    clm_ready(args.clm_endpoint)
    jev = HttpJevClient(api_key=os.environ.get("JEV_LOCAL_API_KEY", "local-shadow"), endpoint=args.jev_endpoint)
    report = run(args.input, args.output, jev, HttpClmClient(endpoint=args.clm_endpoint))
    print(json.dumps(report))
    return 0 if report["status"] == "paired" else 2


if __name__ == "__main__":
    raise SystemExit(main())
