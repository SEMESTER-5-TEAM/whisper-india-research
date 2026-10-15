#!/usr/bin/env python
"""Classify ASR outputs and decompose corpus WER by mutually exclusive error class.

Run from project root:
    python src\classify_errors.py

Input:
    results/lora_generic_hindi/four_condition_per_record.csv

Outputs:
    results/lora_generic_hindi/four_condition_error_class_summary.csv
    results/lora_generic_hindi/four_condition_error_class_by_record.csv
    results/lora_generic_hindi/four_condition_wer_decomposition.png

Priority order: EMPTY -> WRONG_SCRIPT -> RUNAWAY -> ORDINARY.
These are screening categories, not verified hallucination labels.
"""
from pathlib import Path
from collections import Counter
import re
import unicodedata

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "lora_generic_hindi"
INPUT = RESULTS / "four_condition_per_record.csv"


def words(text):
    if pd.isna(text):
        return []
    return str(text).split()


def letter_script_ratio(text):
    letters = [c for c in str(text) if unicodedata.category(c).startswith("L")]
    if not letters:
        return 0.0
    devanagari = sum("\u0900" <= c <= "\u097f" for c in letters)
    return devanagari / len(letters)


def repeated_trigram_fraction(tokens):
    if len(tokens) < 3:
        return 0.0
    trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
    counts = Counter(trigrams)
    repeated_occurrences = sum(count for count in counts.values() if count > 1)
    return repeated_occurrences / len(trigrams)


def classify(row):
    pred = "" if pd.isna(row["prediction_normalized"]) else str(row["prediction_normalized"]).strip()
    ref_tokens = words(row["reference_normalized"])
    pred_tokens = words(pred)
    if not pred:
        return "EMPTY"
    if letter_script_ratio(pred) < 0.50:
        return "WRONG_SCRIPT"
    ratio = len(pred_tokens) / max(1, len(ref_tokens))
    if (len(pred_tokens) >= 12 and ratio >= 1.8) or repeated_trigram_fraction(pred_tokens) >= 0.25:
        return "RUNAWAY"
    return "ORDINARY"


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing {INPUT}. Run src\\score_four_conditions.py first.")
    df = pd.read_csv(INPUT, keep_default_na=False)
    needed = {"condition", "reference_normalized", "prediction_normalized",
              "reference_words", "S", "D", "I", "word_errors"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Input missing columns: {sorted(missing)}")

    df["error_class"] = df.apply(classify, axis=1)
    df["reference_words"] = pd.to_numeric(df["reference_words"], errors="coerce").fillna(0)
    for c in ["S", "D", "I", "word_errors"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    total_ref_words = df.groupby("condition")["reference_words"].sum().to_dict()
    rows = []
    for (condition, category), group in df.groupby(["condition", "error_class"], sort=False):
        denominator = total_ref_words[condition]
        rows.append({
            "condition": condition,
            "error_class": category,
            "recordings": len(group),
            "reference_words_in_class": int(group["reference_words"].sum()),
            "substitutions": int(group["S"].sum()),
            "deletions": int(group["D"].sum()),
            "insertions": int(group["I"].sum()),
            "word_errors": int(group["word_errors"].sum()),
            "WER_contribution_percentage_points": 100 * group["word_errors"].sum() / denominator if denominator else 0,
            "share_of_all_word_errors_percent": 100 * group["word_errors"].sum() / df.loc[df["condition"] == condition, "word_errors"].sum()
                if df.loc[df["condition"] == condition, "word_errors"].sum() else 0,
        })
    summary = pd.DataFrame(rows)
    order = ["EMPTY", "WRONG_SCRIPT", "RUNAWAY", "ORDINARY"]
    summary["error_class"] = pd.Categorical(summary["error_class"], categories=order, ordered=True)
    summary = summary.sort_values(["condition", "error_class"])
    summary["error_class"] = summary["error_class"].astype(str)

    out_summary = RESULTS / "four_condition_error_class_summary.csv"
    out_detail = RESULTS / "four_condition_error_class_by_record.csv"
    out_chart = RESULTS / "four_condition_wer_decomposition.png"
    summary.to_csv(out_summary, index=False)
    df.to_csv(out_detail, index=False)

    # Stacked bars show each mutually exclusive class's contribution to corpus WER.
    pivot = summary.pivot(index="condition", columns="error_class",
                          values="WER_contribution_percentage_points").fillna(0)
    pivot = pivot.reindex(columns=order, fill_value=0)
    pivot = pivot.reindex([
        "baseline_forced_hindi", "lora_forced_hindi",
        "baseline_default", "lora_default"
    ])
    ax = pivot.plot(kind="bar", stacked=True, figsize=(10, 6))
    ax.set_ylabel("Contribution to corpus WER (percentage points)")
    ax.set_xlabel("")
    ax.set_title("Corpus WER decomposition by automatic output class")
    ax.legend(title="Automatic class", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.savefig(out_chart, dpi=180, bbox_inches="tight")
    plt.close()

    print("\nERROR CLASS SUMMARY")
    print(summary[["condition", "error_class", "recordings", "word_errors",
                   "WER_contribution_percentage_points", "share_of_all_word_errors_percent"]]
          .to_string(index=False, formatters={
              "WER_contribution_percentage_points": lambda x: f"{x:.3f}",
              "share_of_all_word_errors_percent": lambda x: f"{x:.2f}",
          }))
    print("\nSaved:")
    print(out_summary.relative_to(ROOT))
    print(out_detail.relative_to(ROOT))
    print(out_chart.relative_to(ROOT))
    print("\nInterpretation note: WRONG_SCRIPT can include Romanized Hindi; RUNAWAY is a heuristic. These labels are not confirmed hallucinations.")


if __name__ == "__main__":
    main()
