"""Publish the woven graph through knitweb's synaptic layer (operator decision: knitweb is the
publication layer, not an ingest tool).

The woven gitnexus graph is compiled into deterministic Fiber synaptic bytecode with
knitweb.synaptic.bytecode.compile_bundle, so identical knowledge always yields identical bytes
and the same digest. Signing needs the originator's private key; it is done by the operator
(knitweb.synaptic.bytecode.sign_bundle) and is deliberately not automated here. Publishing to
OriginTrail (knitweb.synaptic.origintrail) is deferred until bundles are signed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from toddler import _knitweb

_knitweb.load()

from knitweb.synaptic.bytecode import Relation, bundle_digest, compile_bundle, decode_bundle  # noqa: E402

# Code-structure edges stay out of the shared bundle; every other edge kind is shared.
# Deriving the kinds from the graph means a new relation kind can never silently drop out.
PRIVATE_KINDS = frozenset({"defines", "imports", "calls"})


def relations_from_graph(graph: dict) -> list[Relation]:
    source_type = {n["id"]: n["type"] for n in graph["nodes"]}
    return [Relation(e["from"], e["kind"], e["to"], source_type.get(e["from"], "Unknown"))
            for e in graph["edges"] if e["kind"] not in PRIVATE_KINDS]


def build_bundle(graph_path: Path, originator: str, asset_cid: str = "toddler:woven:v1") -> tuple[bytes, str]:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    data = compile_bundle(asset_cid, originator, relations_from_graph(graph))
    decode_bundle(data)  # round-trip check: raises BytecodeError on a malformed bundle
    return data, bundle_digest(data)


def write_bundle(out: Path, data: bytes, sign: Callable[[bytes], str] | None = None) -> Path:
    """Unsigned bundles are written as *.synaptic.unsigned and must not be shared; a relay
    rejects them. With a signing callback (knitweb sign_bundle bound to the operator's key) the
    bundle and its signature are written as *.synaptic and *.synaptic.sig."""
    if sign is None:
        target = out.with_suffix(".synaptic.unsigned")
        target.write_bytes(data)
        return target
    target = out.with_suffix(".synaptic")
    target.write_bytes(data)
    target.with_suffix(".synaptic.sig").write_text(sign(data), encoding="utf-8")
    return target


if __name__ == "__main__":
    woven = Path.home() / ".gitnexus/toddler/woven/toddler_woven_gitnexus.json"
    data, digest = build_bundle(woven, originator="virtuanalytica/toddler")
    path = write_bundle(woven, data)
    print(json.dumps({"bundle": str(path), "bytes": len(data), "digest": digest, "signed": False}, indent=1))
