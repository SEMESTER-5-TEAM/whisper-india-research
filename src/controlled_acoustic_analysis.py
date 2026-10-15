
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results" / "baseline"

df = pd.read_csv(BASE / "acoustic_analysis_data_clean.csv")

# Remove missing predictions and unusable references
df["prediction"] = df["prediction"].fillna("").astype(str).str.strip()
df["reference"] = df["reference"].fillna("").astype(str).str.strip()

df = df[
    df["prediction"].ne("")
    & df["reference"].ne("")
].copy()

df["reference_word_count"] = df["reference"].str.split().str.len()
df = df[df["reference_word_count"] > 0].copy()

# Log-transform word count to reduce the influence of very long transcripts
df["log_reference_words"] = np.log1p(df["reference_word_count"])

# Use a fixed set of acoustic features
features = [
    "silence_ratio_dbfs_40",
    "low_energy_ratio_dbfs_35",
    "overall_rms",
    "mean_zero_crossing_rate",
]

# Standardize acoustic features so coefficients are comparable
for feature in features:
    sd = df[feature].std()
    if sd > 0:
        df[feature + "_z"] = (df[feature] - df[feature].mean()) / sd

print("RECORDS INCLUDED:", len(df))
print("\nCORPUS-LEVEL WER")
# Corpus-level WER is total word edit distance / total reference words.
# This calculation uses the same whitespace-tokenized words as our earlier analysis.
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

for dataset, part in df.groupby("dataset"):
    errors = sum(
        edit_distance(r.split(), h.split())
        for r, h in zip(part["reference"], part["prediction"])
    )
    words = sum(len(r.split()) for r in part["reference"])
    print(
        f"{dataset}: records={len(part)}, "
        f"total reference words={words}, "
        f"corpus WER={errors / words:.4f}"
    )

# Model 1: control for reference length and dataset
model1 = smf.ols(
    "record_wer_clean ~ log_reference_words + C(dataset)",
    data=df,
).fit(cov_type="HC3")

# Model 2: add acoustic features, still controlling for reference length
acoustic_terms = " + ".join(f"{f}_z" for f in features)
formula2 = (
    "record_wer_clean ~ log_reference_words + C(dataset) + "
    + acoustic_terms
)
model2 = smf.ols(formula2, data=df).fit(cov_type="HC3")

print("\nMODEL 1: REFERENCE LENGTH + DATASET")
print(model1.summary())

print("\nMODEL 2: LENGTH + DATASET + ACOUSTIC FEATURES")
print(model2.summary())

out = BASE / "controlled_acoustic_analysis.csv"
df.to_csv(out, index=False)
print("\nSaved analysis data:", out)
