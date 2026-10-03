"""End-to-end check of gui/index.html in a headless browser (run manually; needs playwright).

Set TODDLER_CHROMIUM to a chromium/headless-shell binary if playwright's own is missing."""

import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

from toddler import config

GUI = (Path(__file__).resolve().parents[1] / "gui" / "index.html").as_uri()


def main() -> None:
    exe = os.environ.get("TODDLER_CHROMIUM") or None
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=exe)
        pg = b.new_page(viewport={"width": 1280, "height": 900})
        pg.goto(GUI)
        pg.fill("#r-name", "Data Steward")
        pg.click("[data-add=guardrails]")
        for k, v in {"id": "no-pii-export", "owner": "compliance@acme.nl", "reason": "no pii", "forbidden_tags": "export-pii"}.items():
            pg.fill(f"#guardrails-0-{k}", v)
        pg.click("[data-add=guardrails]")
        for k, v in {"id": "no-pii-export", "owner": "x@y.nl", "reason": "dup", "forbidden_tags": "x"}.items():
            pg.fill(f"#guardrails-1-{k}", v)
        assert "dubbele id" in pg.inner_text("#issues") and pg.is_disabled("#download") and pg.is_disabled("#copy")
        pg.click("#guardrails .item:nth-child(2) .del")
        assert pg.inner_text("#status").strip() == "Configuratie is geldig.", pg.inner_text("#issues")
        d = config.from_dict(json.loads(pg.inner_text("#out")))
        assert d.spec.extra_stop_rules[0].rule_id == "company:no-pii-export"
        m = b.new_page(viewport={"width": 400, "height": 860})
        m.goto(GUI)
        assert not m.evaluate("document.documentElement.scrollWidth > innerWidth")
        b.close()
    print("gui e2e ok")


if __name__ == "__main__":
    main()
