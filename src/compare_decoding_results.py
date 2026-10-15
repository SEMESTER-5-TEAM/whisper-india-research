
import re
import unicodedata
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

BASELINE_FILE = (
    ROOT / "results" / "baseline"
    / "whisper_small_baseline_expanded.csv"
)
INTERVENTION_FILE = (
    ROOT / "results" / "intervention"
    / "whisper_small_no_previous_context.csv"
)

OUT_DIR = ROOT / "results" / "intervention"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize(text):
    """Lowercase, normalize Unicode, remove punctuation, collapse spaces."""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"[^\w\s]|_", " ", text, flags=re.UNICODE)
    return " ".join(text.split())



def edit_distance(a, b):
    """Levenshtein edit distance using a memory-efficient DP."""
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))

    for i, x in enumerate(a, start=1):
        current = [i]
        for j, y in enumerate(b, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (x != y),
                )
            )
        previous = current

    return previous[-1]


def add_metrics(df, prediction_column):
    rows = []

    for _, row in df.iterrows():
        reference = normalize(row["reference"])
        prediction = normalize(row[prediction_column])

        ref_words = reference.split()
        pred_words = prediction.split()

        word_errors = edit_distance(ref_words, pred_words)
        char_errors = edit_distance(reference, prediction)

        rows.append({
            "word_errors": word_errors,
            "reference_words": len(ref_words),
            "char_errors": char_errors,
            "reference_chars": len(reference),
        })

    metrics = pd.DataFrame(rows, index=df.index)
    return pd.concat([df, metrics], axis=1)


def summarize(df, system_name, group_columns):
    rows = []

    for group_key, part in df.groupby(
        group_columns, dropna=False, sort=True
    ):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        result = dict(zip(group_columns, group_key))
        result["system"] = system_name
        result["recordings"] = len(part)

        total_words = part["reference_words"].sum()
        total_chars = part["reference_chars"].sum()

        result["WER"] = (
            part["word_errors"].sum() / total_words
            if total_words else float("nan")
        )
        result["CER"] = (
            part["char_errors"].sum() / total_chars
            if total_chars else float("nan")
        )

        rows.append(result)

    return pd.DataFrame(rows)


def main():
    baseline = pd.read_csv(BASELINE_FILE)
    intervention = pd.read_csv(INTERVENTION_FILE)

    for name, df in [
        ("baseline", baseline),
        ("intervention", intervention),
    ]:
        required = {
            "eval_id", "dataset", "state",
            "reference", "prediction"
        }
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{name} file is missing columns: {sorted(missing)}"
            )

        df["eval_id"] = df["eval_id"].astype(int)

    # Only compare recordings with successful predictions in both systems.
    if "status" in baseline.columns:
        baseline = baseline[
            baseline["status"].astype(str).str.lower().eq("success")
        ].copy()

    if "status" in intervention.columns:
        intervention = intervention[
            intervention["status"].astype(str).str.lower().eq("success")
        ].copy()

    # Check that IDs uniquely identify recordings in both files.
    if baseline["eval_id"].duplicated().any():
        raise ValueError("Duplicate eval_id values in baseline file.")

    if intervention["eval_id"].duplicated().any():
        raise ValueError("Duplicate eval_id values in intervention file.")

    baseline = baseline.rename(
        columns={"prediction": "baseline_prediction"}
    )
    intervention = intervention.rename(
        columns={"prediction": "intervention_prediction"}
    )

    paired = baseline[
        ["eval_id", "dataset", "state", "reference",
         "baseline_prediction"]
    ].merge(
        intervention[
            ["eval_id", "dataset", "state", "reference",
             "intervention_prediction"]
        ],
        on="eval_id",
        how="inner",
        suffixes=("_base", "_int"),
        validate="one_to_one",
    )

    if paired.empty:
        raise ValueError("No matching successful recordings found.")

    # Ensure metadata and references match for every paired recording.
    for col in ["dataset", "state", "reference"]:
        left = paired[f"{col}_base"] if f"{col}_base" in paired else paired[col]
        right = paired[f"{col}_int"] if f"{col}_int" in paired else paired[col]

        if not left.astype(str).equals(right.astype(str)):
            raise ValueError(
                f"Mismatch in {col} between baseline and intervention."
            )

    # Keep a single shared set of metadata/reference columns.
    if "dataset_base" in paired.columns:
        paired["dataset"] = paired["dataset_base"]
        paired["state"] = paired["state_base"]
        paired["reference"] = paired["reference_base"]

    paired = paired[
        ["eval_id", "dataset", "state", "reference",
         "baseline_prediction", "intervention_prediction"]
    ].copy()

    base_metrics = add_metrics(paired, "baseline_prediction")
    int_metrics = add_metrics(paired, "intervention_prediction")

    # Per-recording metric tables, for later paired analysis.
    metric_columns = [
        "word_errors", "reference_words",
        "char_errors", "reference_chars"
    ]

    paired_results = paired[
        ["eval_id", "dataset", "state"]
    ].copy()

    for col in metric_columns:
        paired_results[f"baseline_{col}"] = base_metrics[col].values
        paired_results[f"intervention_{col}"] = int_metrics[col].values

    paired_results.to_csv(
        OUT_DIR / "paired_recording_metrics.csv", index=False
    )

    # Overall and dataset-level corpus WER/CER.
    base_metrics["dataset"] = paired["dataset"]
    int_metrics["dataset"] = paired["dataset"]

    overall = pd.concat([
        summarize(base_metrics, "baseline", ["dataset"]),
        summarize(int_metrics, "no_previous_context", ["dataset"]),
    ], ignore_index=True)

    overall.to_csv(
        OUT_DIR / "comparison_metrics_by_dataset.csv", index=False
    )

    # Region-level results within each dataset.
    base_metrics["state"] = paired["state"]
    int_metrics["state"] = paired["state"]

    regional = pd.concat([
        summarize(base_metrics, "baseline", ["dataset", "state"]),
        summarize(int_metrics, "no_previous_context", ["dataset", "state"]),
    ], ignore_index=True)

    regional.to_csv(
        OUT_DIR / "comparison_metrics_by_region.csv", index=False
    )

    print("\nPAIRED DECODING COMPARISON")
    print("=" * 55)
    print(f"Successful baseline predictions: {len(baseline)}")
    print(f"Successful intervention predictions: {len(intervention)}")
    print(f"Matched recordings compared: {len(paired)}")
    print(
        "Baseline-only recordings:",
        len(set(baseline["eval_id"]) - set(paired["eval_id"]))
    )
    print(
        "Intervention-only recordings:",
        len(set(intervention["eval_id"]) - set(paired["eval_id"]))
    )

    print("\nMETRICS BY DATASET")
    print(
        overall[
            ["dataset", "system", "recordings", "WER", "CER"]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\nSaved:")
    print(OUT_DIR / "paired_recording_metrics.csv")
    print(OUT_DIR / "comparison_metrics_by_dataset.csv")
    print(OUT_DIR / "comparison_metrics_by_region.csv")


if __name__ == "__main__":
    main()
