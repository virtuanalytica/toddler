#!/usr/bin/env python3
"""Assemble and render the Toddler white paper from its maintained sources."""

from __future__ import annotations

import re
from pathlib import Path

from weasyprint import HTML


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/whitepaper"
BASE = DOCS / "body.html"
EXTENSION = DOCS / "work_path_extension.html"
HTML_PATH = DOCS / "toddler-whitepaper-en.html"
PDF_PATH = DOCS / "toddler-whitepaper-en.pdf"


def main() -> None:
    base = BASE.read_text(encoding="utf-8")
    extension = EXTENSION.read_text(encoding="utf-8")
    sections = re.findall(r'<section class="chapter">.*?</section>', extension, re.DOTALL)
    numbered = []
    for section in sections:
        match = re.search(r'<div class="chapter-num">(\d+) ·', section)
        if match is None:
            raise ValueError("extension chapter is missing a number")
        numbered.append((int(match.group(1)), section))
    if sorted(number for number, _ in numbered) != list(range(8, 30)):
        raise ValueError("expected exactly chapters 08 through 29")
    appendix = '<section class="chapter appendix">\n  <div class="chapter-num">30 · Appendix</div>'
    if base.count(appendix) != 1:
        raise ValueError("base paper must contain exactly one chapter 30 appendix")
    body = base.replace(appendix, "\n\n".join(section for _, section in sorted(numbered)) + "\n\n" + appendix)
    old_html = HTML_PATH.read_text(encoding="utf-8")
    prefix, old_body = old_html.split("<body>", 1)
    _, suffix = old_body.split("</body>", 1)
    prefix = prefix.replace(
        'content="Design, objective functions and a brain-network-inspired evaluation method for Toddler, a governed base agent for company roles."',
        'content="Toddler work path, PLS and PAR evidence, and an illustrative five-year business scenario."',
    )
    HTML_PATH.write_text(prefix + "<body>\n" + body.strip() + "\n</body>" + suffix, encoding="utf-8")
    HTML(filename=str(HTML_PATH), base_url=str(ROOT)).write_pdf(str(PDF_PATH))
    print(f"Built {HTML_PATH} and {PDF_PATH} with {len(numbered)} extension chapters")


if __name__ == "__main__":
    main()
