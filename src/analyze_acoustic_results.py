
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results" / "baseline"

FEATURES = BASE / "acoustic_features.csv"
CANDIDATES = BASE / "hallucination_candidates_expanded.csv"
REVIEWS = BASE / "hallucination_review_expanded_completed.csv"

OUT_CORR = BASE / "acoustic_correlations.csv"
OUT_GROUPS = BASE / "acoustic_group_comparisons.csv"
OUT_SUMMARY = BASE / "acoustic_analysis_summary.txt"
OUT_MERGED = BASE / "acoustic_analysis_data.csv"


def main():
    df = pd.read_csv(FEATURES)
    candidates = pd.read_csv(CANDIDATES)
    reviews = pd.read_csv(REVIEWS)

    # Use IDs to avoid depending on row order.
    candidate_ids = set(candidates["eval_id"].dropna().astype(int))
    df["is_candidate"] = df["eval_id"].astype(int).isin(candidate_ids)

    review_labels = (
        reviews[["eval_id", "human_label"]]
        .drop_duplicates("eval_id")
    )
    df = df.merge(review_labels, on="eval_id", how="left")

    # Do not treat missing model outputs as normal predictions.
    df["prediction_missing"] = df["prediction"].isna()
    df["record_wer"] = pd.to_numeric(df["record_wer"], errors="coerce")

    # Keep valid rows for each analysis rather than silently filling values.
    feature_cols = [
        "audio_duration_seconds",
        "silence_ratio_dbfs_40",
        "low_energy_ratio_dbfs_35",
        "overall_rms",
        "mean_zero_crossing_rate",
    ]

    df.to_csv(OUT_MERGED, index=False)

    lines = []
    lines.append("ACOUSTIC ANALYSIS")
    lines.append("=" * 60)
    lines.append(f"Total recordings: {len(df)}")
    lines.append(f"Datasets: {df['dataset'].value_counts().to_dict()}")
    lines.append(f"Missing predictions: {int(df['prediction_missing'].sum())}")
    lines.append(f"Candidate recordings: {int(df['is_candidate'].sum())}")
    lines.append(
        "Reviewed candidate labels: "
        + str(df["human_label"].value_counts(dropna=False).to_dict())
    )
    lines.append(
        "Note: silence features are energy-threshold proxies, not VAD."
    )

    # Spearman correlations, robust to skewed WER distributions.
    corr_rows = []
    for dataset in ["all", "lahaja", "svarah"]:
        subset = df if dataset == "all" else df[df["dataset"] == dataset]

        for feature in feature_cols:
            valid = subset[[feature, "record_wer"]].dropna()
            if len(valid) < 10:
                continue

            rho, p = spearmanr(valid[feature], valid["record_wer"])
            corr_rows.append({
                "dataset": dataset,
                "feature": feature,
                "n": len(valid),
                "spearman_rho": rho,
                "p_value": p,
            })

    corr = pd.DataFrame(corr_rows)
    if not corr.empty:
        corr["p_holm"] = multipletests(
            corr["p_value"].values, method="holm"
        )[1]
        corr.to_csv(OUT_CORR, index=False)

        lines.append("\nSPEARMAN CORRELATIONS WITH RECORD-LEVEL WER")
        lines.append(corr.sort_values("p_holm").to_string(index=False))

    # Compare candidate vs non-candidate acoustic distributions.
    # Candidate status is a heuristic; it is not ground truth.
    group_rows = []
    for dataset in ["all", "lahaja", "svarah"]:
        subset = df if dataset == "all" else df[df["dataset"] == dataset]

        for feature in feature_cols:
            a = subset.loc[
                subset["is_candidate"], feature
            ].dropna().to_numpy()
            b = subset.loc[
                ~subset["is_candidate"], feature
            ].dropna().to_numpy()

            if len(a) < 2 or len(b) < 2:
                continue

            stat, p = mannwhitneyu(a, b, alternative="two-sided")
            group_rows.append({
                "dataset": dataset,
                "feature": feature,
                "candidate_n": len(a),
                "noncandidate_n": len(b),
                "candidate_median": np.median(a),
                "noncandidate_median": np.median(b),
                "median_difference": np.median(a) - np.median(b),
                "mann_whitney_u": stat,
                "p_value": p,
            })

    groups = pd.DataFrame(group_rows)
    if not groups.empty:
        groups["p_holm"] = multipletests(
            groups["p_value"].values, method="holm"
        )[1]
        groups.to_csv(OUT_GROUPS, index=False)

        lines.append("\nACOUSTIC FEATURES: CANDIDATE VS NON-CANDIDATE")
        lines.append(groups.sort_values("p_holm").to_string(index=False))

    # Dataset summaries, with medians to reduce sensitivity to extreme WER.
    lines.append("\nDATASET SUMMARIES")
    summary = df.groupby("dataset").agg(
        recordings=("eval_id", "count"),
        median_wer=("record_wer", "median"),
        mean_wer=("record_wer", "mean"),
        median_duration=("audio_duration_seconds", "median"),
        median_silence_ratio=("silence_ratio_dbfs_40", "median"),
        median_low_energy_ratio=("low_energy_ratio_dbfs_35", "median"),
        candidate_rate=("is_candidate", "mean"),
        missing_prediction_rate=("prediction_missing", "mean"),
    )
    lines.append(summary.to_string())

    OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print("\nSaved:")
    print(OUT_MERGED)
    print(OUT_CORR)
    print(OUT_GROUPS)
    print(OUT_SUMMARY)


if __name__ == "__main__":
    main()
