
import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

BASELINE_FILE = (
    ROOT / "results" / "baseline"
    / "whisper_small_baseline_expanded.csv"
)
INTERVENTION_FILE = (
    ROOT / "results" / "intervention"
    / "whisper_small_no_previous_context.csv"
)
REVIEW_FILE = (
    ROOT / "results" / "baseline"
    / "hallucination_review_expanded_completed.csv"
)

OUT_DIR = ROOT / "results" / "intervention"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def tokens(text):
    return str(text).split()


def repetition_ratio(text, n=2):
    t = tokens(text)
    if len(t) < n:
        return 0.0

    ngrams = [
        tuple(t[i:i + n])
        for i in range(len(t) - n + 1)
    ]

    return 1.0 - len(set(ngrams)) / len(ngrams)


def longest_run(text):
    t = tokens(text)
    if not t:
        return 0

    longest = current = 1

    for i in range(1, len(t)):
        if t[i] == t[i - 1]:
            current += 1
            longest = max(longest, current)
        else:
            current = 1

    return longest


def add_candidate_signals(df, prediction_col):
    result = df.copy()

    result["reference_tokens"] = (
        result["reference"].fillna("").apply(
            lambda x: len(tokens(x))
        )
    )
    result["prediction_tokens"] = (
        result[prediction_col].fillna("").apply(
            lambda x: len(tokens(x))
        )
    )

    result["length_ratio"] = (
        (result["prediction_tokens"] + 1)
        / (result["reference_tokens"] + 1)
    )

    result["bigram_repetition"] = (
        result[prediction_col].fillna("")
        .apply(repetition_ratio)
    )

    result["longest_repeated_run"] = (
        result[prediction_col].fillna("")
        .apply(longest_run)
    )

    result["candidate_reason"] = ""

    result.loc[
        result["longest_repeated_run"] >= 5,
        "candidate_reason"
    ] += "long_repetition;"

    result.loc[
        (result["length_ratio"] >= 3.0)
        & (result["bigram_repetition"] >= 0.50),
        "candidate_reason"
    ] += "length_plus_repetition;"

    result.loc[
        (result["reference_tokens"] <= 5)
        & (result["prediction_tokens"] >= 30),
        "candidate_reason"
    ] += "short_reference_long_prediction;"

    result["candidate"] = result["candidate_reason"].ne("")
    return result


def main():
    baseline = pd.read_csv(BASELINE_FILE)
    intervention = pd.read_csv(INTERVENTION_FILE)

    for name, df in [
        ("baseline", baseline),
        ("intervention", intervention),
    ]:
        required = {
            "eval_id", "dataset", "state",
            "reference", "prediction", "status"
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{name} file missing columns: {sorted(missing)}"
            )

    baseline = baseline[
        baseline["status"].astype(str).str.lower().eq("success")
    ].copy()

    intervention = intervention[
        intervention["status"].astype(str).str.lower().eq("success")
    ].copy()

    baseline["eval_id"] = baseline["eval_id"].astype(int)
    intervention["eval_id"] = intervention["eval_id"].astype(int)

    if baseline["eval_id"].duplicated().any():
        raise ValueError("Duplicate baseline eval_id values.")
    if intervention["eval_id"].duplicated().any():
        raise ValueError("Duplicate intervention eval_id values.")

    # Match identical recordings across both systems.
    paired = baseline[
        ["eval_id", "dataset", "state", "reference", "prediction"]
    ].rename(columns={"prediction": "baseline_prediction"}).merge(
        intervention[
            ["eval_id", "dataset", "state", "reference", "prediction"]
        ].rename(columns={"prediction": "intervention_prediction"}),
        on="eval_id",
        how="inner",
        suffixes=("_base", "_int"),
        validate="one_to_one",
    )

    if paired.empty:
        raise ValueError("No matched recordings found.")

    # Check metadata and references are identical for each pair.
    for column in ["dataset", "state", "reference"]:
        left = paired[f"{column}_base"].astype(str)
        right = paired[f"{column}_int"].astype(str)
        if not left.equals(right):
            raise ValueError(
                f"Mismatch in {column} across systems."
            )

    paired["dataset"] = paired["dataset_base"]
    paired["state"] = paired["state_base"]
    paired["reference"] = paired["reference_base"]

    base_signals = add_candidate_signals(
        paired, "baseline_prediction"
    )
    int_signals = add_candidate_signals(
        paired, "intervention_prediction"
    )

    # Paired candidate status and transition.
    result = paired[
        ["eval_id", "dataset", "state", "reference",
         "baseline_prediction", "intervention_prediction"]
    ].copy()

    result["baseline_candidate"] = base_signals["candidate"].values
    result["intervention_candidate"] = int_signals["candidate"].values
    result["baseline_reason"] = base_signals["candidate_reason"].values
    result["intervention_reason"] = int_signals["candidate_reason"].values

    result["transition"] = "neither_flagged"
    result.loc[
        result["baseline_candidate"]
        & result["intervention_candidate"],
        "transition"
    ] = "flagged_both"

    result.loc[
        result["baseline_candidate"]
        & ~result["intervention_candidate"],
        "transition"
    ] = "baseline_only"

    result.loc[
        ~result["baseline_candidate"]
        & result["intervention_candidate"],
        "transition"
    ] = "intervention_only"

    result.to_csv(
        OUT_DIR / "paired_candidate_comparison.csv",
        index=False,
        encoding="utf-8"
    )

    # Overall candidate rates and transitions.
    n = len(result)
    base_count = int(result["baseline_candidate"].sum())
    int_count = int(result["intervention_candidate"].sum())

    print("=" * 65)
    print("PAIRED HALLUCINATION-CANDIDATE COMPARISON")
    print("=" * 65)
    print(f"Matched recordings: {n}")
    print(f"Baseline candidates: {base_count} ({100*base_count/n:.2f}%)")
    print(f"Intervention candidates: {int_count} ({100*int_count/n:.2f}%)")
    print(f"Candidate count change: {int_count - base_count:+d}")
    print(
        "Candidate rate change (percentage points): "
        f"{100*(int_count-base_count)/n:+.2f}"
    )

    print("\nPAIRED TRANSITIONS")
    print(result["transition"].value_counts().to_string())

    print("\nBY DATASET")
    rows = []
    for dataset, part in result.groupby("dataset", sort=True):
        b = int(part["baseline_candidate"].sum())
        i = int(part["intervention_candidate"].sum())
        rows.append({
            "dataset": dataset,
            "recordings": len(part),
            "baseline_candidates": b,
            "baseline_rate_pct": 100 * b / len(part),
            "intervention_candidates": i,
            "intervention_rate_pct": 100 * i / len(part),
            "count_change": i - b,
            "rate_change_pp": 100 * (i - b) / len(part),
        })

    by_dataset = pd.DataFrame(rows)
    print(
        by_dataset.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )
    by_dataset.to_csv(
        OUT_DIR / "candidate_comparison_by_dataset.csv",
        index=False,
        encoding="utf-8"
    )

    print("\nBY DATASET AND STATE")
    regional_rows = []
    for (dataset, state), part in result.groupby(
        ["dataset", "state"], sort=True, dropna=False
    ):
        b = int(part["baseline_candidate"].sum())
        i = int(part["intervention_candidate"].sum())
        regional_rows.append({
            "dataset": dataset,
            "state": state,
            "recordings": len(part),
            "baseline_candidates": b,
            "baseline_rate_pct": 100 * b / len(part),
            "intervention_candidates": i,
            "intervention_rate_pct": 100 * i / len(part),
            "count_change": i - b,
            "rate_change_pp": 100 * (i - b) / len(part),
        })

    by_region = pd.DataFrame(regional_rows)
    by_region.to_csv(
        OUT_DIR / "candidate_comparison_by_region.csv",
        index=False,
        encoding="utf-8"
    )

    print("\nLargest regional candidate-rate reductions:")
    print(
        by_region.sort_values("rate_change_pp").head(10).to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    # Compare computed baseline flags with the saved original candidate list.
    original_candidates_file = (
        ROOT / "results" / "baseline"
        / "hallucination_candidates_expanded.csv"
    )
    if original_candidates_file.exists():
        original = pd.read_csv(original_candidates_file)
        original_ids = set(original["eval_id"].astype(int))
        computed_ids = set(
            result.loc[result["baseline_candidate"], "eval_id"].astype(int)
        )
        print("\nBASELINE RULE CONSISTENCY CHECK")
        print(f"Saved original candidate IDs: {len(original_ids)}")
        print(f"Recomputed baseline candidate IDs: {len(computed_ids)}")
        print(f"IDs differing: {len(original_ids ^ computed_ids)}")

    # Show how the baseline manual-review labels distribute across
    # transitions. This does not establish intervention labels.
    if REVIEW_FILE.exists():
        reviews = pd.read_csv(REVIEW_FILE)
        if {"eval_id", "human_label"}.issubset(reviews.columns):
            reviewed = result.merge(
                reviews[["eval_id", "human_label"]],
                on="eval_id",
                how="left",
                validate="one_to_one"
            )
            print("\nBASELINE REVIEW LABELS BY TRANSITION")
            print(
                pd.crosstab(
                    reviewed["transition"],
                    reviewed["human_label"].fillna("NOT_REVIEWED")
                ).to_string()
            )

    print("\nSaved files:")
    print(OUT_DIR / "paired_candidate_comparison.csv")
    print(OUT_DIR / "candidate_comparison_by_dataset.csv")
    print(OUT_DIR / "candidate_comparison_by_region.csv")


if __name__ == "__main__":
    main()
