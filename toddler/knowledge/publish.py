"""Publish the woven graph through knitweb's synaptic layer (operator decision: knitweb is the
publication layer, not an ingest tool).

The woven gitnexus graph is compiled into deterministic Fiber synaptic bytecode with
knitweb.synaptic.bytecode.compile_bundle, so identical knowledge always yields identical bytes
and the same digest. Signing needs the originator's private key; it is done by the operator
(knitweb.synaptic.bytecode.sign_bundle) and is deliberately not automated here.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_KNITWEB_SRC = os.environ.get("TODDLER_KNITWEB_SRC", "/media/knight2/EDS2/projects/knitweb/src")
if _KNITWEB_SRC not in sys.path:
    sys.path.append(_KNITWEB_SRC)

from knitweb.synaptic.bytecode import Relation, bundle_digest, compile_bundle, decode_bundle  # noqa: E402

GRAPH_KINDS = ("is_precursor_of", "is_part_of", "is_superseded_by", "has_risk", "serves", "is_problem_for",
               "uses", "is_input_of", "is_complemented_by", "used_in_toddler", "states", "implemented_in",
               "rests_on", "motivates")


def relations_from_graph(graph: dict) -> list[Relation]:
    source_type = {n["id"]: n["type"] for n in graph["nodes"]}
    return [Relation(e["from"], e["kind"], e["to"], source_type.get(e["from"], "Unknown"))
            for e in graph["edges"] if e["kind"] in GRAPH_KINDS]


def build_bundle(graph_path: Path, originator: str, asset_cid: str = "toddler:woven:v1") -> tuple[bytes, str]:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    data = compile_bundle(asset_cid, originator, relations_from_graph(graph))
    decode_bundle(data)  # round-trip check: raises BytecodeError on a malformed bundle
    return data, bundle_digest(data)


if __name__ == "__main__":
    woven = Path.home() / ".gitnexus/toddler/woven/toddler_woven_gitnexus.json"
    data, digest = build_bundle(woven, originator="virtuanalytica/toddler")
    out = woven.with_suffix(".synaptic")
    out.write_bytes(data)
    print(json.dumps({"bundle": str(out), "bytes": len(data), "digest": digest}, indent=1))
