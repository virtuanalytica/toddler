#!/usr/bin/env python3
"""Illustrative Preschooler adoption and unit-cost scenarios, never observed sales."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/businessplan/preschooler-scenarios.json"
YEARS = (2027, 2028, 2029, 2030, 2031)
BASE_OWNERS = (2, 5, 12, 25, 60)
BASE_AGENTS = (10, 50, 250, 1000, 5000)
ASSUMPTIONS = {
    "owner_platform_eur_per_year": 6000,
    "active_agent_platform_eur_per_year": 240,
    "accepted_tasks_per_agent_year": 600,
    "electricity_eur_per_kwh": 0.30,
    "local_system_kwh_per_attempt": 0.010,
    "local_acceptance_rate": 0.80,
    "hardware_and_storage_eur_per_attempt": 0.004,
    "training_and_evaluation_budget_eur": 50000,
    "annual_accepted_tasks_for_training_amortization": 300000,
    "cloud_reference_eur_per_accepted_task": 0.015,
}


def adoption_rows(scale: float) -> list[dict]:
    rows = []
    for year, owners, agents in zip(YEARS, BASE_OWNERS, BASE_AGENTS):
        owners, agents = round(owners * scale), round(agents * scale)
        # This is platform licence value served by Preschooler, not incremental company revenue.
        attributed_licence = (owners * ASSUMPTIONS["owner_platform_eur_per_year"]
                              + agents * ASSUMPTIONS["active_agent_platform_eur_per_year"])
        rows.append({"year": year, "paying_owners": owners,
                     "average_active_preschooler_agents": agents,
                     "accepted_tasks": agents * ASSUMPTIONS["accepted_tasks_per_agent_year"],
                     "attributed_platform_licence_eur": attributed_licence})
    return rows


def unit_cost() -> dict:
    a = ASSUMPTIONS
    variable_per_attempt = (a["local_system_kwh_per_attempt"] * a["electricity_eur_per_kwh"]
                            + a["hardware_and_storage_eur_per_attempt"])
    variable_per_accepted = variable_per_attempt / a["local_acceptance_rate"]
    amortized = a["training_and_evaluation_budget_eur"] / a["annual_accepted_tasks_for_training_amortization"]
    return {"electricity_eur_per_attempt": round(a["local_system_kwh_per_attempt"] * a["electricity_eur_per_kwh"], 6),
            "variable_eur_per_accepted_task": round(variable_per_accepted, 6),
            "training_eur_per_accepted_task_at_assumed_volume": round(amortized, 6),
            "total_eur_per_accepted_task_at_assumed_volume": round(variable_per_accepted + amortized, 6),
            "cloud_reference_eur_per_accepted_task": a["cloud_reference_eur_per_accepted_task"]}


def main() -> None:
    data = {
        "status": "planning_assumptions_not_observed_results",
        "currency": "nominal EUR excluding VAT",
        "assumptions": ASSUMPTIONS,
        "adoption": {"downside": adoption_rows(0.5), "base": adoption_rows(1.0),
                     "upside": adoption_rows(1.5)},
        "unit_cost": unit_cost(),
        "limitations": [
            "Attributed licence value is a subset of the existing Toddler platform scenario, not additive revenue.",
            "The cloud reference is an illustrative input and must be replaced with measured comparable invoices.",
            "Review labour, networking, support, insurance and general operating expense are excluded from unit cost.",
            "The assumed training budget and accepted-task volume are independent; low volume raises amortized cost.",
            "A 5,000-agent 2031 base case is a hypothesis, not observed demand or committed contracts.",
        ],
    }
    OUTPUT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
