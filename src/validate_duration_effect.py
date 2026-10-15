
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results" / "baseline"

df = pd.read_csv(BASE / "acoustic_analysis_data_clean.csv")

# Exclude missing or blank predictions
df["prediction"] = df["prediction"].fillna("").astype(str).str.strip()
df = df[df["prediction"].ne("")].copy()

# Reference transcript length
df["reference_word_count"] = (
    df["reference"].fillna("").astype(str).str.split().str.len()
)
df = df[df["reference_word_count"] > 0].copy()

print("REFERENCE LENGTH CHECK")
print(
    df.groupby("dataset")["reference_word_count"]
    .agg(["count", "median", "mean", "min", "max"])
    .to_string()
)

print("\nDURATION VS REFERENCE LENGTH")
for dataset in ["all", "lahaja", "svarah"]:
    sub = df if dataset == "all" else df[df["dataset"] == dataset]
    rho, p = spearmanr(
        sub["audio_duration_seconds"], sub["reference_word_count"]
    )
    print(f"{dataset.upper()}: rho={rho:.3f}, p={p:.4g}")

print("\nDURATION VS WER WITHIN REFERENCE-LENGTH GROUPS")
for dataset in ["all", "lahaja", "svarah"]:
    sub = df if dataset == "all" else df[df["dataset"] == dataset]
    sub = sub.copy()

    # Separate short, medium, and long references
    sub["length_group"] = pd.qcut(
        sub["reference_word_count"],
        q=3,
        labels=["short", "medium", "long"],
        duplicates="drop",
    )

    print(f"\n{dataset.upper()}")
    for group, part in sub.groupby("length_group", observed=True):
        if len(part) < 20:
            continue
        rho, p = spearmanr(
            part["audio_duration_seconds"], part["record_wer_clean"]
        )
        print(
            f"{group}: n={len(part)}, "
            f"rho={rho:.3f}, p={p:.4g}"
        )

out = BASE / "duration_effect_validation.csv"
df.to_csv(out, index=False)
print("\nSaved:", out)
