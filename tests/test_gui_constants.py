"""The GUI hand-copies base rule and question ids; this test keeps them in sync with Python."""

import re
from pathlib import Path

from toddler import fastpath, stop

GUI = (Path(__file__).resolve().parents[1] / "gui" / "index.html").read_text(encoding="utf-8")


def _block(name: str) -> str:
    m = re.search(rf"const {name} = \[(.*?)\];", GUI, re.S)
    assert m, f"{name} not found in gui/index.html"
    return m.group(1)


def test_base_rule_ids_match_stop_defaults():
    ids = re.findall(r'\["([a-z-]+)",', _block("BASE_RULES"))
    assert ids == [r.rule_id for r in stop.DEFAULT_RULES]


def test_base_question_ids_match_fastpath_defaults():
    ids = re.findall(r'"([a-z_]+)"', _block("BASE_QUESTIONS"))
    assert ids == [q.qid for q in fastpath.DEFAULT_QUESTIONS]
