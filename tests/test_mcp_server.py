import json

from toddler.learn import mcp_server as S


def test_initialize_echoes_protocol_and_lists_tools():
    r = S.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}})
    assert r["result"]["protocolVersion"] == "2025-03-26" and "tools" in r["result"]["capabilities"]
    names = {t["name"] for t in S.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]}
    assert names == {"list_toddlers", "list_tasks", "play_episode", "iq_probe", "trace"}


def test_notifications_get_no_reply_and_unknown_methods_an_error():
    assert S.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert S.handle({"jsonrpc": "2.0", "id": 3, "method": "nope"})["error"]["code"] == -32601


def test_tool_results_and_tool_errors_are_reported_not_raised():
    ok = S.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "list_tasks"}})
    assert not ok["result"]["isError"] and "doorkey8" in json.loads(ok["result"]["content"][0]["text"])
    bad = S.handle({"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "explode"}})
    assert bad["result"]["isError"] and "unknown tool" in bad["result"]["content"][0]["text"]
