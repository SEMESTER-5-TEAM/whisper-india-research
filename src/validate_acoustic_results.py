
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results" / "baseline"

df = pd.read_csv(BASE / "acoustic_analysis_data.csv")

# Treat blank or whitespace-only predictions as missing.
pred = df["prediction"].fillna("").astype(str).str.strip()
df["prediction_missing"] = pred.eq("")

# Recalculate WER only for usable predictions.
def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        current = [i]
        for j, y in enumerate(b, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (x != y),
            ))
        previous = current
    return previous[-1]

def calculate_wer(reference, prediction):
    ref = str(reference).split()
    hyp = str(prediction).split()
    if not ref:
        return np.nan
    return edit_distance(ref, hyp) / len(ref)

df["record_wer_clean"] = np.nan
valid = ~df["prediction_missing"]

df.loc[valid, "record_wer_clean"] = [
    calculate_wer(r, h)
    for r, h in zip(
        df.loc[valid, "reference"],
        df.loc[valid, "prediction"],
    )
]

print("RECORD COUNTS")
print("Total:", len(df))
print("Missing predictions:", int(df["prediction_missing"].sum()))
print("Usable predictions:", int(valid.sum()))

print("\nCLEAN WER SUMMARY")
print(
    df.loc[valid]
    .groupby("dataset")["record_wer_clean"]
    .agg(["count", "median", "mean", "std"])
    .to_string()
)

features = [
    "audio_duration_seconds",
    "silence_ratio_dbfs_40",
    "low_energy_ratio_dbfs_35",
    "overall_rms",
    "mean_zero_crossing_rate",
]

print("\nSPEARMAN CORRELATIONS WITH CLEAN WER")
for dataset in ["all", "lahaja", "svarah"]:
    subset = df.loc[valid] if dataset == "all" else df.loc[
        valid & df["dataset"].eq(dataset)
    ]

    print(f"\n{dataset.upper()} (n={len(subset)})")
    for feature in features:
        pair = subset[[feature, "record_wer_clean"]].dropna()
        rho, p = spearmanr(pair[feature], pair["record_wer_clean"])
        print(
            f"{feature}: rho={rho:.3f}, "
            f"p={p:.4g}, n={len(pair)}"
        )

out = BASE / "acoustic_analysis_data_clean.csv"
df.to_csv(out, index=False)
print("\nSaved clean analysis data to:", out)
