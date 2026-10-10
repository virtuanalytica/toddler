#!/usr/bin/env python3
"""Render the reproducible G0–G4 technical paper to styled HTML and PDF."""

from __future__ import annotations

import re
from pathlib import Path

import markdown
from weasyprint import HTML


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/technicalpaper"
SOURCE = DOCS / "TECHNICAL_PAPER_G0_G4.md"
DEST_HTML = DOCS / "technical-paper-g0-g4.html"
DEST_PDF = DOCS / "technical-paper-g0-g4.pdf"

CSS = """
@page {
  size: A4;
  margin: 18mm 17mm 18mm 17mm;
  background: #f7f9f7;
  @top-left { content: "VIRTUANALYTICA  /  TODDLER"; color: #53747c; font: 8pt "DejaVu Sans"; letter-spacing: 1pt; }
  @bottom-left { content: "Technical paper · G0–G4 · 11 oktober 2026"; color: #53747c; font: 8pt "DejaVu Sans"; }
  @bottom-right { content: counter(page); color: #53747c; font: 9pt "DejaVu Sans"; }
}
@page:first { @top-left { content: ""; } @bottom-left { content: ""; } @bottom-right { content: ""; } }
* { box-sizing: border-box; }
body { font: 9.4pt/1.51 "DejaVu Sans", sans-serif; color: #173244; }
.hero { min-height: 255mm; padding: 4mm 0; position: relative; }
.hero .brand img { width: 100%; max-height: 32mm; margin: 0 0 36mm; }
.hero h1 { font: bold 34pt/1.1 "DejaVu Sans", sans-serif; color: #112b3d; letter-spacing: -1pt; border-left: 7pt solid #2da78f; padding-left: 12pt; margin-bottom: 18mm; }
.hero p { color: #486878; font-size: 11pt; line-height: 1.6; }
.hero .mark img { width: 70%; max-height: 100mm; margin: 25mm auto 0; }
h2 { break-before: page; break-after: avoid; color: #102d40; font: bold 18pt/1.2 "DejaVu Sans", sans-serif; border-bottom: 2pt solid #69c6b1; padding-bottom: 7pt; margin: 0 0 13pt; }
h3 { color: #176277; font-size: 12pt; break-after: avoid; }
p { margin: 0 0 9pt; orphans: 3; widows: 3; }
strong { color: #112b3d; }
code { font: 8pt/1.35 "DejaVu Sans Mono", monospace; background: #e8f1ee; padding: 1pt 2pt; overflow-wrap: anywhere; }
pre { background: #122e3c; color: #edf7f3; padding: 12pt; font-size: 7.6pt; line-height: 1.35; white-space: pre-wrap; overflow-wrap: anywhere; break-inside: avoid; }
pre code { color: inherit; background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 12pt 0 16pt; font-size: 7.4pt; }
th { background: #153e4f; color: #fff; padding: 6pt; text-align: left; }
td { border-bottom: .5pt solid #c8d9d4; padding: 5pt 6pt; vertical-align: top; overflow-wrap: anywhere; }
tr:nth-child(even) td { background: #eaf2ef; }
tr { break-inside: avoid; }
ul, ol { padding-left: 18pt; }
li { margin-bottom: 4pt; }
a { color: #087784; text-decoration: none; overflow-wrap: anywhere; }
main img { display: block; max-width: 100%; max-height: 130mm; object-fit: contain; margin: 13pt auto 18pt; }
blockquote { border-left: 4pt solid #c99b4b; padding: 10pt; background: #eff3e9; }
@media screen { body { width: 210mm; margin: auto; padding: 18mm; background: #f7f9f7; box-shadow: 0 0 18px #cbd7d2; } }
"""


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    required = (
        "## 7. G4", "## 8. Cognitie", "## 10. Datamodel",
        "## 13. Trainingsprotocol", "## 14. Ontwikkelaars",
        "## Appendix A.", "## Appendix B.", "## Appendix C.",
    )
    missing = [item for item in required if item not in source]
    if missing:
        raise ValueError(f"technical paper incomplete: {missing}")
    html_body = markdown.markdown(source, extensions=["tables", "fenced_code"])
    cover, body = html_body.split("<h2>Abstract</h2>", 1)
    cover = re.sub(r"<p>(<img[^>]*virtuanalytica-logo[^>]*>)</p>", r'<div class="brand">\1</div>', cover, count=1)
    cover = re.sub(r"<p>(<img[^>]*toddler-mark[^>]*>)</p>", r'<div class="mark">\1</div>', cover, count=1)
    page = (
        '<!DOCTYPE html><html lang="nl"><head><meta charset="utf-8">'
        '<title>Toddler · technische paper G0–G4</title>'
        f'<style>{CSS}</style></head><body><section class="hero">{cover}</section>'
        f'<main><h2>Abstract</h2>{body}</main></body></html>'
    )
    DEST_HTML.write_text(page, encoding="utf-8")
    HTML(filename=str(DEST_HTML), base_url=str(DOCS)).write_pdf(str(DEST_PDF))
    print(f"Built {DEST_HTML} and {DEST_PDF}")


if __name__ == "__main__":
    main()
