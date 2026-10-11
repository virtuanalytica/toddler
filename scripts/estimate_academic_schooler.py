#!/usr/bin/env python3
"""Transparent planning arithmetic for eight-domain 4B and 32B curricula."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/businessplan/academic-schooler-capacity.json"


def estimate() -> dict:
    domains = 8
    pilot_examples = 10_000
    pilot_tokens = 1536
    target_examples = 250_000
    target_tokens = 2048
    specialist_epochs = 2
    consolidation_epochs = 2
    pilot_exposures = domains * pilot_examples * pilot_tokens * specialist_epochs
    one_full_pass = domains * target_examples * target_tokens
    specialist_exposures = one_full_pass * specialist_epochs
    consolidation_exposures = one_full_pass * consolidation_epochs
    total_exposures = specialist_exposures + consolidation_exposures
    # NVIDIA Qwen2.5-32B LoRA DGX-H100 reference: 2,250.31 tokens/s/GPU, eight GPUs.
    reference_cluster_tps = 8 * 2250.31
    realized_tps = [reference_cluster_tps * x for x in (0.35, 0.70)]
    hours_low = total_exposures / max(realized_tps) / 3600 * 1.25
    hours_high = total_exposures / min(realized_tps) / 3600 * 1.25
    gpu_hours_low, gpu_hours_high = 8 * hours_low, 8 * hours_high
    teacher_pool = 10
    candidates_per_item = 4
    full_roster_calibration_fraction = 0.05
    mean_candidates = candidates_per_item + full_roster_calibration_fraction * (teacher_pool - candidates_per_item)
    generated_tokens = round(domains * target_examples * mean_candidates * 512)
    exhaustive_tokens = domains * target_examples * teacher_pool * 512
    generation_hours = [generated_tokens / tps / 3600 for tps in (1000, 300)]
    raw_text_gb = one_full_pass * 4 / 1e9
    candidates_gb = generated_tokens * 4 / 1e9
    token_cache_gb = one_full_pass * 4 / 1e9
    metadata_gb = domains * target_examples * 2000 / 1e9
    structured_gb = raw_text_gb + candidates_gb + token_cache_gb + metadata_gb
    training_low_with_reserve = round(gpu_hours_low * 3.20 * 1.5)
    training_high_with_reserve = round(gpu_hours_high * 6.16 * 1.5)
    generation_low = round(generation_hours[0] * 8 * 3.20)
    generation_high = round(generation_hours[1] * 8 * 3.20)
    cloud_storage_three_months = round(3 * (4000 * 0.219 + 8000 * 0.05))
    return {
        "status": "planning_estimate_not_measured_toddler_training",
        "domains": domains,
        "dataset_design": {
            "pilot_examples_per_domain": pilot_examples,
            "pilot_tokens_per_example": pilot_tokens,
            "pilot_epochs_per_domain": specialist_epochs,
            "pilot_token_exposures": pilot_exposures,
            "target_examples_per_domain": target_examples,
            "target_tokens_per_example": target_tokens,
            "target_examples_total": domains * target_examples,
            "specialist_epochs_per_domain": specialist_epochs,
            "consolidation_epochs": consolidation_epochs,
            "specialist_token_exposures": specialist_exposures,
            "consolidation_token_exposures": consolidation_exposures,
            "total_32b_token_exposures": total_exposures,
        },
        "compute": {
            "reference": "NVIDIA NeMo AutoModel Qwen2.5-32B LoRA DGX-H100; proxy, not measured Qwen3-32B",
            "reference_tokens_per_second_per_gpu": 2250.31,
            "gpu_count": 8,
            "effective_fraction_of_reference": [0.35, 0.70],
            "evaluation_overhead_fraction": 0.25,
            "cluster_hours_range": [round(hours_low), round(hours_high)],
            "gpu_hours_range": [round(gpu_hours_low), round(gpu_hours_high)],
            "training_usd_range_at_3_20_to_6_16_per_gpu_hour": [round(gpu_hours_low * 3.20), round(gpu_hours_high * 6.16)],
            "training_usd_range_with_50pct_rerun_reserve": [training_low_with_reserve, training_high_with_reserve],
            "pilot_single_ada_assumed_tps_range": [300, 1000],
            "pilot_hours_range_at_assumed_tps": [round(pilot_exposures / 1000 / 3600), round(pilot_exposures / 300 / 3600)],
        },
        "generation": {
            "teacher_pool_size": teacher_pool,
            "regular_candidates_per_example": candidates_per_item,
            "full_roster_calibration_fraction": full_roster_calibration_fraction,
            "mean_candidates_per_example": mean_candidates,
            "output_tokens_per_candidate": 512,
            "output_tokens_total": generated_tokens,
            "exhaustive_ten_per_item_output_tokens": exhaustive_tokens,
            "assumed_cluster_tps_range": [300, 1000],
            "cluster_hours_range": [round(generation_hours[0]), round(generation_hours[1])],
            "eight_h100_usd_range_at_3_20_per_gpu_hour": [generation_low, generation_high],
            "note": "Proxy for sequential or leased teacher generation, not a measured ten-model throughput; prompt, routing, cold-load and failure costs excluded.",
        },
        "storage": {
            "ten_pinned_teacher_snapshots_gb_estimate": 73.6,
            "teacher_snapshot_note": "HF file-metadata preflight on 2026-10-11; manifests and caches add overhead, and later revisions change the total.",
            "raw_text_gb_at_4_bytes_per_token": round(raw_text_gb, 1),
            "candidate_text_gb_at_4_bytes_per_token": round(candidates_gb, 1),
            "token_cache_gb_int32": round(token_cache_gb, 1),
            "provenance_metadata_gb_at_2kb_per_item": round(metadata_gb, 1),
            "structured_data_subtotal_gb": round(structured_gb, 1),
            "source_snapshots_tb_range": [0.5, 2.0],
            "checkpoints_and_revisions_tb_range": [0.3, 0.8],
            "required_fast_free_nvme_tb_recommended": 4,
            "archive_or_backup_tb_recommended": 8,
            "cloud_storage_three_months_usd_before_included_capacity": cloud_storage_three_months,
        },
        "review": {
            "expert_review_fraction_range": [0.01, 0.05],
            "items_range": [int(domains * target_examples * 0.01), int(domains * target_examples * 0.05)],
            "eur_per_item_range": [10, 20],
            "budget_eur_range": [int(domains * target_examples * 0.01 * 10), int(domains * target_examples * 0.05 * 20)],
            "note": "Only valid if the remaining items have independent objective verification; open-ended items need broader expert review.",
        },
        "combined_external_budget": {
            "compute_and_three_month_cloud_storage_usd_range": [
                training_low_with_reserve + generation_low + cloud_storage_three_months,
                training_high_with_reserve + generation_high + cloud_storage_three_months,
            ],
            "expert_review_eur_range": [int(domains * target_examples * 0.01 * 10),
                                        int(domains * target_examples * 0.05 * 20)],
            "excludes": ["dataset rights", "staff", "taxes", "customer acceptance", "local hardware acquisition"],
        },
        "limits": [
            "4B cannot become 32B by repeated fine-tuning; the 32B phase starts from a separate licensed base.",
            "Qwen2.5 throughput is a proxy for Qwen3 and must be replaced by a short paid calibration run.",
            "Eight domain adapters do not automatically make one integrated checkpoint; balanced consolidation is budgeted separately.",
            "Generation proxy assumes four candidate outputs per item plus all ten on 5% calibration items; ten on every item is a separate upper-cost scenario.",
            "Teacher models share several base families; count independent error patterns, not only model IDs.",
            "Training GPU costs exclude storage, dataset rights, expert review, orchestration, taxes and customer validation.",
        ],
    }


def main() -> None:
    data = estimate()
    OUTPUT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
