#!/usr/bin/env python3
"""Build the standalone, scenario-checked Preschooler business-concept PDF."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import markdown
from weasyprint import HTML

from build_business_plan import CSS
from forecast_preschooler import OUTPUT, main as build_forecast

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/businessplan"
SOURCE = PLAN / "PRESCHOOLER_4B_BUSINESS_CONCEPT.md"
HTML_OUT = PLAN / "preschooler-4b-business-concept.html"
PDF_OUT = PLAN / "preschooler-4b-business-concept.pdf"


def euro(amount: float) -> str:
    return "€" + f"{amount:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")


def render_source(source: str, data: dict) -> str:
    rows = data["adoption"]["base"]
    header = "| Jaar | Betalende eigenaren | Actieve Preschooler-agenten | Geaccepteerde taken | Toegerekende licentiewaarde |\n| --- | ---: | ---: | ---: | ---: |"
    table = "\n".join([header] + [f"| {r['year']} | {r['paying_owners']:,} | {r['average_active_preschooler_agents']:,} | {r['accepted_tasks']:,} | {euro(r['attributed_platform_licence_eur'])} |" for r in rows])
    replacements = {
        "ADOPTION_TABLE": table,
        "DOWNSIDE_AGENTS": f"{data['adoption']['downside'][-1]['average_active_preschooler_agents']:,}",
        "DOWNSIDE_VALUE": euro(data["adoption"]["downside"][-1]["attributed_platform_licence_eur"]),
        "UPSIDE_AGENTS": f"{data['adoption']['upside'][-1]['average_active_preschooler_agents']:,}",
        "UPSIDE_VALUE": euro(data["adoption"]["upside"][-1]["attributed_platform_licence_eur"]),
        "VARIABLE_COST": euro(data["unit_cost"]["variable_eur_per_accepted_task"]),
        "TRAINING_COST": euro(data["unit_cost"]["training_eur_per_accepted_task_at_assumed_volume"]),
        "TOTAL_COST": euro(data["unit_cost"]["total_eur_per_accepted_task_at_assumed_volume"]),
    }
    for key, value in replacements.items():
        token = "{{" + key + "}}"
        if source.count(token) != 1:
            raise ValueError(f"Expected exactly one {token}")
        source = source.replace(token, value)
    if "{{" in source:
        raise ValueError("Unresolved template placeholder")
    return source


def main() -> None:
    build_forecast()
    data = json.loads(OUTPUT.read_text(encoding="utf-8"))
    if data["status"] != "planning_assumptions_not_observed_results":
        raise ValueError("Scenario status is not hypothetical")
    body = render_source(SOURCE.read_text(encoding="utf-8"), data)
    content = markdown.markdown(body, extensions=["tables", "fenced_code"])
    css = CSS.replace("VIRTUANALYTICA  /  TODDLER", "VIRTUANALYTICA  /  PRESCHOOLER").replace(
        "Ondernemingsplan · scenario, 11 oktober 2026", "Businessconcept · scenario, 11 oktober 2026")
    css += "\nbody > p:first-child img {width: 100%; max-height: 28mm; margin: 4mm 0 26mm;}\n"
    css += "\nbody > p:nth-of-type(3) img {width: 100%; max-height: 85mm; margin: 9mm auto;}\n"
    title = escape("Virtuanalytica · Preschooler-4B · businessconcept")
    HTML_OUT.write_text(f'<!DOCTYPE html><html lang="nl"><head><meta charset="utf-8"><title>{title}</title><style>{css}</style></head><body>{content}</body></html>', encoding="utf-8")
    HTML(filename=str(HTML_OUT), base_url=str(PLAN)).write_pdf(str(PDF_OUT))
    print(f"Built {HTML_OUT} and {PDF_OUT}")


if __name__ == "__main__":
    main()
