#!/usr/bin/env python3
"""Render the source business plan and a forecast-derived vector chart."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import markdown
from weasyprint import HTML


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/businessplan"
SOURCE = PLAN / "BUSINESS_PLAN_2027_2031.md"
DEST_HTML = PLAN / "business-plan-2027-2031.html"
DEST_PDF = PLAN / "business-plan-2027-2031.pdf"
CHART = PLAN / "assets/financial-path.svg"
FORECAST = ROOT / "docs/whitepaper/forecast_2027_2031.json"


def chart_svg(rows: list[dict]) -> str:
    width, height = 1000, 390
    max_revenue = max(row["revenue_eur"] for row in rows)
    scale = 240 / max_revenue
    bars = []
    for index, row in enumerate(rows):
        x = 130 + index * 175
        revenue = row["revenue_eur"]
        ebitda = row["ebitda_eur"]
        bar_height = revenue * scale
        ebitda_y = 292 - ebitda * scale
        label = f"{revenue / 1_000_000:.2f}"
        bars.append(f'<rect x="{x}" y="{292-bar_height:.1f}" width="72" height="{bar_height:.1f}" rx="6" fill="#30a78e"/>')
        bars.append(f'<text x="{x+36}" y="{277-bar_height:.1f}" fill="#16374e" font-size="17" text-anchor="middle">€{label}m</text>')
        bars.append(f'<circle cx="{x+36}" cy="{ebitda_y:.1f}" r="8" fill="#d99d40"/>')
        bars.append(f'<text x="{x+36}" y="330" fill="#16374e" font-size="18" text-anchor="middle">{row["year"]}</text>')
    line = " ".join(f'{130+i*175+36},{292-row["ebitda_eur"]*scale:.1f}' for i, row in enumerate(rows))
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="t d">
<title id="t">Illustratieve platformomzet en EBITDA 2027–2031</title>
<desc id="d">Groene omzetstaven en gouden EBITDA-lijn. Het zijn scenarioaannames, geen gerealiseerde cijfers.</desc>
<rect width="{width}" height="{height}" rx="20" fill="#eff5f2"/>
<text x="42" y="43" fill="#153b53" font-family="DejaVu Sans, sans-serif" font-size="25" font-weight="700">Platformomzet en EBITDA</text>
<text x="42" y="68" fill="#47677a" font-family="DejaVu Sans, sans-serif" font-size="16">scenario in miljoenen euro · geen gerealiseerde omzet</text>
<path d="M80 292 H960" stroke="#94adaf" stroke-width="2"/>
<polyline points="{line}" fill="none" stroke="#d99d40" stroke-width="4"/>
<g font-family="DejaVu Sans, sans-serif">{''.join(bars)}</g>
<rect x="730" y="26" width="18" height="18" fill="#30a78e"/><text x="756" y="42" fill="#153b53" font-family="DejaVu Sans, sans-serif" font-size="14">omzet</text>
<circle cx="854" cy="35" r="8" fill="#d99d40"/><text x="872" y="42" fill="#153b53" font-family="DejaVu Sans, sans-serif" font-size="14">EBITDA</text>
</svg>'''


CSS = """
@page {
  size: A4;
  margin: 20mm 19mm 19mm 19mm;
  background: #f8f8f3;
  @top-right { content: "VIRTUANALYTICA  /  TODDLER"; color: #58737d; font: 8pt "DejaVu Sans"; letter-spacing: 1pt; }
  @bottom-right { content: counter(page); color: #58737d; font: 9pt "DejaVu Sans"; }
  @bottom-left { content: "Ondernemingsplan · scenario, 11 oktober 2026"; color: #58737d; font: 8pt "DejaVu Sans"; }
}
@page:first { @top-right { content: ""; } @bottom-right { content: ""; } @bottom-left { content: ""; } }
* { box-sizing: border-box; }
body { font-family: "DejaVu Sans", sans-serif; color: #173044; font-size: 9.6pt; line-height: 1.48; }
h1 { color: #102338; font-size: 31pt; line-height: 1.1; letter-spacing: -1.3pt; border-left: 7pt solid #30a78e; padding-left: 12pt; margin: 28pt 0 18pt; }
h2 { color: #102338; font-size: 18pt; line-height: 1.18; margin: 0 0 13pt; padding-bottom: 8pt; border-bottom: 2pt solid #78ddc7; break-before: page; break-after: avoid; }
h3 { color: #14516b; font-size: 12pt; break-after: avoid; }
p { margin: 0 0 9pt; orphans: 3; widows: 3; }
strong { color: #102338; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.5pt; background: #e8f0eb; padding: 1pt 2pt; overflow-wrap: anywhere; }
a { color: #126a72; text-decoration: none; overflow-wrap: anywhere; }
img { display: block; max-width: 100%; max-height: 116mm; margin: 12pt auto 18pt; object-fit: contain; }
body > p:first-child img { width: 100%; margin: 4mm 0 34mm; }
body > p:nth-of-type(3) img { width: 74%; margin: 28mm auto 10mm; }
table { border-collapse: collapse; width: 100%; font-size: 8pt; margin: 13pt 0 15pt; page-break-inside: avoid; }
th { text-align: left; background: #14354d; color: #fff; padding: 7pt 6pt; }
td { padding: 6pt; border-bottom: .5pt solid #c9d7d3; vertical-align: top; }
tr:nth-child(even) td { background: #edf4f0; }
ul, ol { padding-left: 18pt; margin-bottom: 10pt; }
li { margin-bottom: 4pt; }
blockquote { border-left: 4pt solid #d6a44f; padding: 10pt 13pt; background: #f0f3e9; color: #315269; }
@media screen { body { width: 210mm; padding: 19mm; margin: auto; background: #f8f8f3; box-shadow: 0 0 20px #cbd7d2; } }
"""


def main() -> None:
    forecast = json.loads(FORECAST.read_text(encoding="utf-8"))
    rows = forecast["base"]
    assert len(rows) == 5 and rows[-1]["revenue_eur"] == 15_840_000
    CHART.write_text(chart_svg(rows), encoding="utf-8")
    source = SOURCE.read_text(encoding="utf-8")
    required = ("15,840", "16.000", "9.600.000", "€4.300.350", "Knitweb", "ChemField")
    missing = [token for token in required if token not in source]
    if missing:
        raise ValueError(f"business plan missing expected model facts: {missing}")
    content = markdown.markdown(source, extensions=["tables", "fenced_code"])
    title = escape("Virtuanalytica · Toddler-agenteneconomie")
    html = f'<!DOCTYPE html><html lang="nl"><head><meta charset="utf-8"><title>{title}</title><style>{CSS}</style></head><body>{content}</body></html>'
    DEST_HTML.write_text(html, encoding="utf-8")
    HTML(filename=str(DEST_HTML), base_url=str(PLAN)).write_pdf(str(DEST_PDF))
    print(f"Built {DEST_HTML} and {DEST_PDF}")


if __name__ == "__main__":
    main()
