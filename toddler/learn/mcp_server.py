"""Toddler as an MCP server (stdio, JSON-RPC 2.0), so a chat harness can put a real toddler in front of
its models: OMP (`~/.omp/agent/mcp.json`), Claude Code, or any MCP client.

Dependency-free on purpose: the MCP stdio transport is newline-delimited JSON-RPC, and the Python SDK
pulls a starlette/pydantic stack that conflicts with jevserver's fastapi pin. Only stdout carries the
protocol; everything else goes to stderr.

Tools (all computed by toddler code, never narrated by a model):
  list_toddlers   generations in the register with their members and selection verdicts
  list_tasks      tasks with environment ids and solve thresholds
  play_episode    let one toddler play one level; returns the trajectory as text frames
  iq_probe        IQ quotient of one toddler against the frozen G1 reference
  trace           ancestry and descendants of a toddler, or the full trace of a derived agent

    python3 -m toddler.learn.mcp_server
"""

from __future__ import annotations

import json
import sys
import traceback

PROTOCOL = "2025-06-18"
SERVER = {"name": "toddler", "version": "0.2.0"}

TOOLS = [
    {"name": "list_toddlers", "description": "List toddler generations in the register: members, roles and the "
     "Darwin selection verdict (survived / extinct / control) of each generation.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "list_tasks", "description": "List the benchmark tasks a toddler can be tested on (environment id, "
     "solve threshold, whether it is a MiniGrid level).", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "play_episode", "description": "Let one toddler play one level of a task and return what it did: "
     "actions, return, score normalised to 0 = random and 1 = solved, and text frames (MiniGrid: ASCII grid, "
     "the toddler is the arrow). Use a seed the toddler has not seen, e.g. 400000-999999.",
     "inputSchema": {"type": "object", "required": ["toddler", "task", "seed"], "properties": {
         "toddler": {"type": "string", "description": "generation/toddler_id, e.g. G1/t1001"},
         "task": {"type": "string", "description": "task name from list_tasks, e.g. doorkey5"},
         "seed": {"type": "integer", "description": "level seed"},
         "mode": {"type": "string", "enum": ["greedy", "sample"], "default": "greedy"}}}},
    {"name": "iq_probe", "description": "Measure one toddler's IQ quotient against the frozen G1 reference "
     "(100 = a typical G1 toddler) on 30 held-out levels of the four reference tasks. Takes about a minute.",
     "inputSchema": {"type": "object", "required": ["toddler"], "properties": {
         "toddler": {"type": "string", "description": "generation/toddler_id, e.g. G1/t1001"}}}},
    {"name": "trace", "description": "Trace evolution: for a toddler (generation/id) its ancestors with the "
     "inherited modules and descendants; for an agent (agent:<id>) the chain of derivations and "
     "transmorphoses back to every Toddler foundation.",
     "inputSchema": {"type": "object", "required": ["ref"], "properties": {
         "ref": {"type": "string", "description": "G2/t2001 or agent:<id>"}}}},
]


def _ledger():
    from toddler.learn import lineage as L
    from toddler.learn import play

    return L.Ledger(play.registry_root())


def call(name: str, args: dict) -> object:
    from toddler.learn import play
    from toddler.learn import tasks as T

    if name == "list_tasks":
        return {t: {"env_id": v.env_id, "solved": v.solved, "grid": v.grid} for t, v in T.TASKS.items()}
    if name == "list_toddlers":
        led, out = _ledger(), {}
        for b in led._events("born", "backfilled"):
            g = b["ref"].split("/", 1)[0]
            out.setdefault(g, {"members": [], "verdict": None})["members"].append(
                {"toddler": b["ref"], "role": b["role"], "inherits_from": [i["parent"] for i in b["inherits"]]})
        for g in out:
            v = led.verdict(g)
            out[g]["verdict"] = v["verdict"] if v else "pending"
        return out
    if name == "play_episode":
        return play.episode(args["toddler"], args["task"], int(args["seed"]), args.get("mode", "greedy"))
    if name == "iq_probe":
        return play.iq_probe(args["toddler"])
    if name == "trace":
        led, ref = _ledger(), args["ref"]
        if ref.startswith("agent:"):
            return led.trace_agent(ref[len("agent:"):])
        if not led.birth(ref):
            raise KeyError(f"{ref} is not in the evolution ledger")
        return {"toddler": led.birth(ref), "ancestry": led.ancestry(ref), "descendants": led.descendants(ref)}
    raise KeyError(f"unknown tool {name!r}")


def handle(msg: dict) -> dict | None:
    mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
    if mid is None:                      # notification (e.g. notifications/initialized): no reply
        return None
    if method == "initialize":
        result = {"protocolVersion": params.get("protocolVersion", PROTOCOL), "capabilities": {"tools": {}},
                  "serverInfo": SERVER}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        try:
            out = call(params.get("name", ""), params.get("arguments") or {})
            result = {"content": [{"type": "text", "text": json.dumps(out, indent=1, default=str)}], "isError": False}
        except Exception as exc:          # a tool failure is reported to the model, never a crash
            traceback.print_exc(file=sys.stderr)
            result = {"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}], "isError": True}
    else:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            reply = handle(json.loads(line))
        except json.JSONDecodeError as exc:
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"parse error: {exc}"}}
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
