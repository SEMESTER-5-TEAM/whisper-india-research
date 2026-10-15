#!/usr/bin/env python
"""Paired speaker-cluster bootstrap for four-condition LAHAJA results.

Run from project root:
    python src\speaker_cluster_bootstrap.py

Uses the per-record scoring and automatic class file already produced.
Resamples test speaker IDs with replacement and includes all records for each
sampled speaker, including repeated clusters. All comparisons are paired.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "lora_generic_hindi"
INPUT = RESULTS / "four_condition_error_class_by_record.csv"
N_BOOT = 2000
SEED = 42
CONDITIONS = [
    "baseline_forced_hindi", "lora_forced_hindi",
    "baseline_default", "lora_default",
]
PAIRS = [
    ("forced_hindi_lora_minus_baseline", "lora_forced_hindi", "baseline_forced_hindi"),
    ("default_lora_minus_baseline", "lora_default", "baseline_default"),
]
CLASSES = ["EMPTY", "WRONG_SCRIPT", "RUNAWAY", "ORDINARY"]


def corpus_wer(group):
    denom = group["reference_words"].sum()
    return group["word_errors"].sum() / denom if denom else np.nan


def class_contribution(group, category):
    denom = group["reference_words"].sum()
    errors = group.loc[group["error_class"] == category, "word_errors"].sum()
    return 100.0 * errors / denom if denom else np.nan


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing {INPUT}; run scoring and classification first.")
    df = pd.read_csv(INPUT, keep_default_na=False)
    required = {"condition", "sp_id", "reference_words", "word_errors", "error_class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input missing columns: {sorted(missing)}")
    df["reference_words"] = pd.to_numeric(df["reference_words"], errors="coerce").fillna(0)
    df["word_errors"] = pd.to_numeric(df["word_errors"], errors="coerce").fillna(0)
    df["sp_id"] = df["sp_id"].astype(str)
    df = df[df["condition"].isin(CONDITIONS)].copy()

    # The speaker and recording cohorts must match across all four conditions.
    speaker_sets = {}
    key_sets = {}
    for condition in CONDITIONS:
        sub = df[df["condition"] == condition]
        if sub.empty:
            raise ValueError(f"No rows for condition {condition}")
        speaker_sets[condition] = set(sub["sp_id"])
        key_cols = [c for c in ["source_shard", "source_row"] if c in sub.columns]
        if len(key_cols) == 2:
            key_sets[condition] = set(map(tuple, sub[key_cols].astype(str).to_numpy()))
    if len({frozenset(x) for x in speaker_sets.values()}) != 1:
        raise ValueError("Speaker sets differ between conditions; refusing paired bootstrap.")
    if key_sets and len({frozenset(x) for x in key_sets.values()}) != 1:
        raise ValueError("Recording key sets differ between conditions; refusing paired bootstrap.")

    speakers = sorted(next(iter(speaker_sets.values())))
    rng = np.random.default_rng(SEED)
    draws = []

    # Pre-split condition data for faster resampling.
    by_condition_speaker = {
        (cond, sp): df[(df["condition"] == cond) & (df["sp_id"] == sp)]
        for cond in CONDITIONS for sp in speakers
    }

    for b in range(N_BOOT):
        sampled = rng.choice(speakers, size=len(speakers), replace=True)
        # Preserve duplicate sampled clusters by concatenating each occurrence.
        resampled = {}
        for cond in CONDITIONS:
            parts = [by_condition_speaker[(cond, sp)] for sp in sampled]
            resampled[cond] = pd.concat(parts, ignore_index=True)
        row = {"replicate": b}
        for cond, grp in resampled.items():
            row[f"WER::{cond}"] = corpus_wer(grp)
            for category in CLASSES:
                row[f"CLASS::{cond}::{category}"] = class_contribution(grp, category)
        for label, lora_cond, base_cond in PAIRS:
            row[f"DIFF::{label}"] = row[f"WER::{lora_cond}"] - row[f"WER::{base_cond}"]
        draws.append(row)

    boot = pd.DataFrame(draws)
    rows = []
    for cond in CONDITIONS:
        full = df[df["condition"] == cond]
        rows.append({
            "metric": f"corpus_WER::{cond}",
            "estimate": corpus_wer(full),
            "ci_2_5": boot[f"WER::{cond}"].quantile(0.025),
            "ci_97_5": boot[f"WER::{cond}"].quantile(0.975),
            "unit": "WER fraction",
        })
        for category in CLASSES:
            rows.append({
                "metric": f"WER_contribution_percentage_points::{cond}::{category}",
                "estimate": class_contribution(full, category),
                "ci_2_5": boot[f"CLASS::{cond}::{category}"].quantile(0.025),
                "ci_97_5": boot[f"CLASS::{cond}::{category}"].quantile(0.975),
                "unit": "percentage points",
            })
    for label, lora_cond, base_cond in PAIRS:
        point = corpus_wer(df[df["condition"] == lora_cond]) - corpus_wer(df[df["condition"] == base_cond])
        rows.append({
            "metric": f"paired_WER_difference::{label}",
            "estimate": point,
            "ci_2_5": boot[f"DIFF::{label}"].quantile(0.025),
            "ci_97_5": boot[f"DIFF::{label}"].quantile(0.975),
            "unit": "WER fraction; negative favours LoRA",
        })

    summary = pd.DataFrame(rows)
    summary_path = RESULTS / "speaker_cluster_bootstrap_summary.csv"
    draws_path = RESULTS / "speaker_cluster_bootstrap_replicates.csv"
    json_path = RESULTS / "speaker_cluster_bootstrap_metadata.json"
    summary.to_csv(summary_path, index=False)
    boot.to_csv(draws_path, index=False)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({
            "replicates": N_BOOT,
            "seed": SEED,
            "resampling_unit": "speaker ID, sampled with replacement",
            "number_of_speakers": len(speakers),
            "paired_comparisons": [x[0] for x in PAIRS],
            "confidence_interval": "percentile 95% interval (2.5th, 97.5th percentiles)",
            "note": "WER is calculated from summed edit errors divided by summed reference words in each resampled cluster sample."
        }, f, indent=2)

    print(f"Speakers: {len(speakers)} | bootstrap replicates: {N_BOOT} | seed: {SEED}")
    print("\nPAIRED WER DIFFERENCES (negative favours LoRA)")
    print(summary[summary["metric"].str.startswith("paired_WER_difference")].to_string(
        index=False, formatters={c: lambda x: f"{x:.4f}" for c in ["estimate", "ci_2_5", "ci_97_5"]}))
    print("\nCORPUS WER WITH SPEAKER-CLUSTER 95% CIs")
    print(summary[summary["metric"].str.startswith("corpus_WER::")].to_string(
        index=False, formatters={c: lambda x: f"{x:.4f}" for c in ["estimate", "ci_2_5", "ci_97_5"]}))
    print("\nSaved:")
    print(summary_path.relative_to(ROOT))
    print(draws_path.relative_to(ROOT))
    print(json_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
