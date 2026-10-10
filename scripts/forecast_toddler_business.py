#!/usr/bin/env python3
"""Reproduce the illustrative 2027–2031 owner-led paid-agent scenario.

Amounts are nominal EUR excluding VAT. Gross agent work value is customer/owner
GMV, not Toddler revenue. This is a planning model, not observed demand.
"""

from __future__ import annotations

import json
from pathlib import Path


YEARS = (2027, 2028, 2029, 2030, 2031)
BASE = {
    "paying_owners": (5, 12, 25, 50, 100),
    "average_active_paid_agents": (25, 150, 800, 4_000, 16_000),
    "partner_owned_robot_cells": (0, 0, 2, 8, 25),
    "operating_expense_eur": (700_000, 1_000_000, 1_800_000, 3_300_000, 6_500_000),
    "capital_expenditure_eur": (150_000, 200_000, 300_000, 500_000, 800_000),
}
PRICE = {
    "owner_eur_per_year": 6_000,
    "agent_platform_eur_per_year": 240,
    "accepted_jobs_per_agent_month": 50,
    "accepted_job_gross_eur": 10,
    "marketplace_take_rate": 0.10,
    "robot_cell_eur_per_year": 72_000,
}
DIRECT_COST_SHARE = {
    "owner": 0.20,
    "agent_platform": 0.30,
    "marketplace": 0.35,
    "robot_cell": 0.40,
}

MONTHLY_2027_OWNERS = (2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8)
MONTHLY_2027_AGENTS = (5, 9, 12, 16, 19, 23, 27, 30, 34, 37, 43, 45)


def calculate_monthly_2027() -> list[dict]:
    """Illustrative cash calendar; receipts lag invoices by one month."""
    assert sum(MONTHLY_2027_OWNERS) == 60
    assert sum(MONTHLY_2027_AGENTS) == 300
    rows = []
    previous_invoice_cents = 0
    closing_cents = 0
    monthly_opex_cents, remainder = divmod(70_000_000, 12)
    for index, (owners, agents) in enumerate(zip(MONTHLY_2027_OWNERS, MONTHLY_2027_AGENTS)):
        invoice_cents = owners * 50_000 + agents * 7_000
        direct_cost_cents = owners * 10_000 + agents * 2_350
        opex_cents = monthly_opex_cents + (index < remainder)
        capex_cents = 7_500_000 if index in (0, 6) else 0
        net_cents = previous_invoice_cents - direct_cost_cents - opex_cents - capex_cents
        closing_cents += net_cents
        rows.append({
            "month": index + 1,
            "paying_owners": owners,
            "active_paid_agents": agents,
            "invoiced_eur": invoice_cents / 100,
            "receipts_eur": previous_invoice_cents / 100,
            "direct_cost_eur": direct_cost_cents / 100,
            "operating_expense_eur": opex_cents / 100,
            "capital_expenditure_eur": capex_cents / 100,
            "net_cash_eur": net_cents / 100,
            "closing_cash_before_financing_eur": closing_cents / 100,
        })
        previous_invoice_cents = invoice_cents
    assert sum(row["invoiced_eur"] for row in rows) == 51_000
    assert sum(row["direct_cost_eur"] for row in rows) == 13_050
    assert round(sum(row["operating_expense_eur"] for row in rows), 2) == 700_000
    assert rows[-1]["closing_cash_before_financing_eur"] == -819_200
    return rows


def calculate(scale: float = 1.0, opex_scale: float = 1.0) -> list[dict]:
    rows = []
    cumulative_cash = 0
    for index, year in enumerate(YEARS):
        owners = round(BASE["paying_owners"][index] * scale)
        agents = round(BASE["average_active_paid_agents"][index] * scale)
        cells = round(BASE["partner_owned_robot_cells"][index] * scale)
        accepted_jobs = agents * PRICE["accepted_jobs_per_agent_month"] * 12
        agent_gmv = accepted_jobs * PRICE["accepted_job_gross_eur"]
        revenue = {
            "owner": owners * PRICE["owner_eur_per_year"],
            "agent_platform": agents * PRICE["agent_platform_eur_per_year"],
            "marketplace": round(agent_gmv * PRICE["marketplace_take_rate"]),
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
            "paying_owners": owners,
            "average_active_paid_agents": agents,
            "paid_agents_per_owner": round(agents / owners, 1) if owners else None,
            "accepted_jobs": accepted_jobs,
            "agent_gmv_eur": agent_gmv,
            "partner_owned_robot_cells": cells,
            "owner_revenue_eur": revenue["owner"],
            "agent_platform_revenue_eur": revenue["agent_platform"],
            "marketplace_revenue_eur": revenue["marketplace"],
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
    base = calculate()
    output = {
        "status": "illustrative_assumptions_not_observed_results",
        "units": "nominal EUR, excluding VAT",
        "business_model": "customer_economically_owns_agent_portfolio; platform_licenses_and_takes_fee",
        "price": PRICE,
        "direct_cost_share": DIRECT_COST_SHARE,
        "base": base,
        "monthly_2027": calculate_monthly_2027(),
        "downside_2031": calculate(scale=0.5)[-1],
        "upside_2031": calculate(scale=1.5, opex_scale=1.2)[-1],
        "funding_trough_eur": -min(row["cumulative_cash_proxy_eur"] for row in base),
        "limitations": [
            "Agent GMV belongs to the customer/owner; only licence fees and marketplace commission are platform revenue.",
            "Average active paid agents are full-year equivalents, not cumulative created agents or bank IBANs.",
            "Each agent has an internal ledger; dedicated IBAN rollout requires bank approval and is not assumed.",
            "A 10-euro job must be batched for settlement; per-job card fees could invalidate marketplace margin.",
            "Cash proxy is EBITDA minus capex; excludes tax, interest, depreciation, financing and working capital.",
            "Monthly 2027 cash illustrates a one-month receipt delay and no opening capital; it excludes VAT, taxes, debt and PSP reserves.",
            "No token appreciation, speculative token sales or hardware sales are booked.",
            "Robot cells are partner or customer owned; the fee covers software and service only.",
        ],
    }
    destination = Path(__file__).resolve().parents[1] / "docs/whitepaper/forecast_2027_2031.json"
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
