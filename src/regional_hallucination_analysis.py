import pandas as pd
from pathlib import Path

# ============================================================
# FILES
# ============================================================

BASELINE_FILE = Path(
    "results/baseline/whisper_small_baseline.csv"
)

REVIEW_FILE = Path(
    "results/baseline/hallucination_review_completed.csv"
)

OUTPUT_DIR = Path("results/baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "regional_hallucination_analysis.csv"


# ============================================================
# LOAD FILES
# ============================================================

baseline = pd.read_csv(BASELINE_FILE)
review = pd.read_csv(REVIEW_FILE)

print("=" * 70)
print("REGIONAL HALLUCINATION ANALYSIS")
print("=" * 70)

print(f"Baseline records: {len(baseline)}")
print(f"Reviewed candidates: {len(review)}")


# ============================================================
# CHECK REVIEW LABELS
# ============================================================

print("\nHuman review labels:")
print(review["human_label"].value_counts(dropna=False))


# ============================================================
# MERGE REVIEW WITH BASELINE
# ============================================================

# The review file should contain eval_id.
review_small = review[
    ["eval_id", "human_label"]
].copy()

review_small["eval_id"] = review_small["eval_id"].astype(int)

baseline["eval_id"] = baseline["eval_id"].astype(int)

merged = baseline.merge(
    review_small,
    on="eval_id",
    how="left"
)


# ============================================================
# CREATE REVIEW STATUS
# ============================================================

merged["review_status"] = merged["human_label"].fillna(
    "NOT_REVIEWED"
)


# ============================================================
# REGIONAL ANALYSIS
# ============================================================

rows = []

for (dataset, state), group in merged.groupby(
    ["dataset", "state"],
    dropna=False
):

    total = len(group)

    hallucinations = (
        group["human_label"] == "HALLUCINATION"
    ).sum()

    ordinary_errors = (
        group["human_label"] == "ORDINARY_ASR_ERROR"
    ).sum()

    wrong_language = (
        group["human_label"] == "WRONG_LANGUAGE"
    ).sum()

    reviewed = (
        group["human_label"].notna()
    ).sum()

    candidates = reviewed

    rows.append({
        "dataset": dataset,
        "state": state,
        "baseline_samples": total,
        "reviewed_candidates": candidates,
        "hallucinations": hallucinations,
        "ordinary_asr_errors": ordinary_errors,
        "wrong_language": wrong_language,
        "hallucination_rate_all_samples": (
            hallucinations / total
            if total > 0 else 0
        ),
        "candidate_rate_all_samples": (
            candidates / total
            if total > 0 else 0
        ),
        "hallucination_fraction_among_reviewed": (
            hallucinations / reviewed
            if reviewed > 0 else 0
        )
    })


regional = pd.DataFrame(rows)

regional = regional.sort_values(
    ["dataset", "hallucination_rate_all_samples"],
    ascending=[True, False]
)


# ============================================================
# SAVE
# ============================================================

regional.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("\n" + "=" * 70)
print("STATE-WISE RESULTS")
print("=" * 70)

print(regional.to_string(index=False))


# ============================================================
# DATASET SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

dataset_summary = []

for dataset, group in merged.groupby("dataset"):

    total = len(group)

    hallucinations = (
        group["human_label"] == "HALLUCINATION"
    ).sum()

    ordinary_errors = (
        group["human_label"] == "ORDINARY_ASR_ERROR"
    ).sum()

    wrong_language = (
        group["human_label"] == "WRONG_LANGUAGE"
    ).sum()

    reviewed = group["human_label"].notna().sum()

    dataset_summary.append({
        "dataset": dataset,
        "baseline_samples": total,
        "reviewed_candidates": reviewed,
        "hallucinations": hallucinations,
        "ordinary_asr_errors": ordinary_errors,
        "wrong_language": wrong_language,
        "hallucination_rate": (
            hallucinations / total
            if total > 0 else 0
        ),
        "candidate_rate": (
            reviewed / total
            if total > 0 else 0
        )
    })

dataset_summary = pd.DataFrame(dataset_summary)

print(dataset_summary.to_string(index=False))


# ============================================================
# OVERALL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("OVERALL SUMMARY")
print("=" * 70)

total = len(merged)

hallucinations = (
    merged["human_label"] == "HALLUCINATION"
).sum()

ordinary_errors = (
    merged["human_label"] == "ORDINARY_ASR_ERROR"
).sum()

wrong_language = (
    merged["human_label"] == "WRONG_LANGUAGE"
).sum()

reviewed = merged["human_label"].notna().sum()

print(f"Total baseline samples       : {total}")
print(f"Reviewed candidates          : {reviewed}")
print(f"Confirmed hallucinations     : {hallucinations}")
print(f"Ordinary ASR errors          : {ordinary_errors}")
print(f"Wrong-language cases         : {wrong_language}")

print(
    f"Hallucination rate           : "
    f"{hallucinations / total:.4%}"
)

print(
    f"Candidate rate               : "
    f"{reviewed / total:.4%}"
)

print(
    f"Hallucination fraction among "
    f"reviewed candidates         : "
    f"{hallucinations / reviewed:.4%}"
)

print("\nSaved:")
print(OUTPUT_FILE.resolve())