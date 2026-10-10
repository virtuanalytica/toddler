#!/usr/bin/env python3
"""Reproduce the illustrative 2027–2031 Toddler business scenario.

All amounts are nominal EUR, excluding VAT. This is a planning model, not
observed revenue, a valuation, or an investment forecast.
"""

from __future__ import annotations

import json
from pathlib import Path


YEARS = (2027, 2028, 2029, 2030, 2031)
BASE = {
    "paying_organizations": (8, 25, 70, 160, 320),
    "billable_verified_tasks": (200_000, 1_000_000, 4_000_000, 12_000_000, 30_000_000),
    "partner_owned_robot_cells": (0, 2, 8, 25, 70),
    "operating_expense_eur": (850_000, 1_500_000, 2_800_000, 5_100_000, 8_500_000),
    "capital_expenditure_eur": (180_000, 300_000, 450_000, 750_000, 1_200_000),
}
PRICE = {
    "organization_eur_per_year": 24_000,
    "verified_task_eur": 0.12,
    "robot_cell_eur_per_year": 72_000,
}
DIRECT_COST_SHARE = {"organization": 0.25, "task": 0.40, "robot_cell": 0.35}


def calculate(scale: float = 1.0, opex_scale: float = 1.0) -> list[dict]:
    rows = []
    cumulative_cash = 0
    for index, year in enumerate(YEARS):
        organizations = round(BASE["paying_organizations"][index] * scale)
        tasks = round(BASE["billable_verified_tasks"][index] * scale)
        cells = round(BASE["partner_owned_robot_cells"][index] * scale)
        revenue = {
            "organization": organizations * PRICE["organization_eur_per_year"],
            "task": round(tasks * PRICE["verified_task_eur"]),
            "robot_cell": cells * PRICE["robot_cell_eur_per_year"],
        }
        total_revenue = sum(revenue.values())
        direct_cost = round(sum(revenue[k] * DIRECT_COST_SHARE[k] for k in revenue))
        gross_profit = total_revenue - direct_cost
        opex = round(BASE["operating_expense_eur"][index] * opex_scale)
        capex = BASE["capital_expenditure_eur"][index]
        ebitda = gross_profit - opex
        cash_proxy = ebitda - capex
        cumulative_cash += cash_proxy
        rows.append({
            "year": year,
            "paying_organizations": organizations,
            "billable_verified_tasks": tasks,
            "partner_owned_robot_cells": cells,
            "organization_revenue_eur": revenue["organization"],
            "task_revenue_eur": revenue["task"],
            "robot_cell_revenue_eur": revenue["robot_cell"],
            "revenue_eur": total_revenue,
            "direct_cost_eur": direct_cost,
            "gross_profit_eur": gross_profit,
            "operating_expense_eur": opex,
            "ebitda_eur": ebitda,
            "capital_expenditure_eur": capex,
            "cash_proxy_eur": cash_proxy,
            "cumulative_cash_proxy_eur": cumulative_cash,
        })
    return rows


def main() -> None:
    output = {
        "status": "illustrative_assumptions_not_observed_results",
        "units": "nominal EUR, excluding VAT",
        "price": PRICE,
        "direct_cost_share": DIRECT_COST_SHARE,
        "base": calculate(),
        "downside_2031": calculate(scale=0.5)[-1],
        "upside_2031": calculate(scale=1.5, opex_scale=1.2)[-1],
        "limitations": [
            "Cash proxy is EBITDA minus capex; excludes tax, interest, depreciation, financing and working capital.",
            "No token appreciation, speculative token sales or hardware sales are booked.",
            "Robot cells are partner or customer owned; the fee covers software and service only.",
        ],
    }
    destination = Path(__file__).resolve().parents[1] / "docs/whitepaper/forecast_2027_2031.json"
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
