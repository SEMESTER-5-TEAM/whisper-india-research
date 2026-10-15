import re
import unicodedata
import pandas as pd
from pathlib import Path


BASE = Path("results/baseline")
PREDICTIONS = BASE / "whisper_small_baseline_expanded.csv"
REVIEWS = BASE / "hallucination_review_expanded_completed.csv"


def normalize(text):
    """Lowercase, normalize Unicode, remove punctuation, collapse spaces."""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"[^\w\s]|_", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def edit_distance(a, b):
    """Levenshtein edit distance using a memory-efficient DP row."""
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current

    return previous[-1]


def calculate_metrics(group):
    total_word_errors = 0
    total_reference_words = 0
    total_char_errors = 0
    total_reference_chars = 0

    for _, row in group.iterrows():
        ref = normalize(row["reference"])
        pred = normalize(row["prediction"])

        ref_words = ref.split()
        pred_words = pred.split()

        total_word_errors += edit_distance(ref_words, pred_words)
        total_reference_words += len(ref_words)

        total_char_errors += edit_distance(ref, pred)
        total_reference_chars += len(ref)

    return pd.Series({
        "recordings": len(group),
        "reference_words": total_reference_words,
        "WER": (
            total_word_errors / total_reference_words
            if total_reference_words else float("nan")
        ),
        "reference_characters": total_reference_chars,
        "CER": (
            total_char_errors / total_reference_chars
            if total_reference_chars else float("nan")
        ),
    })


def main():
    df = pd.read_csv(PREDICTIONS)

    # Use successful inference rows only.
    df = df[df["status"].eq("success")].copy()

    df["reference"] = df["reference"].fillna("")
    df["prediction"] = df["prediction"].fillna("")

    # Overall and dataset-level corpus metrics.
    overall = calculate_metrics(df).to_frame().T
    overall.insert(0, "group", "ALL_DATASETS")

    by_dataset = (
        df.groupby("dataset", dropna=False)
        .apply(calculate_metrics, include_groups=False)
        .reset_index()
        .rename(columns={"dataset": "group"})
    )

    # Regional metrics, kept separate for each dataset.
    by_region = (
        df.groupby(["dataset", "state"], dropna=False)
        .apply(calculate_metrics, include_groups=False)
        .reset_index()
    )

    out_overall = pd.concat([overall, by_dataset], ignore_index=True)
    out_overall.to_csv(
        BASE / "normalized_metrics_by_dataset.csv", index=False
    )
    by_region.to_csv(
        BASE / "normalized_metrics_by_region.csv", index=False
    )

    # Summarize manually reviewed candidate labels separately.
    reviews = pd.read_csv(REVIEWS)
    print("\nMANUAL REVIEW LABEL COUNTS")
    print(reviews["human_label"].value_counts(dropna=False).to_string())

    confirmed = reviews[
        reviews["human_label"].eq("HALLUCINATION")
    ].copy()

    print("\nCONFIRMED HALLUCINATION CANDIDATES BY DATASET")
    print(
        confirmed.groupby("dataset").size().to_string()
    )

    print("\nNORMALIZED METRICS: OVERALL AND BY DATASET")
    print(out_overall.round(4).to_string(index=False))

    print("\nFILES SAVED:")
    print(BASE / "normalized_metrics_by_dataset.csv")
    print(BASE / "normalized_metrics_by_region.csv")
    print(
        "\nNote: candidate-review counts are not total hallucination "
        "prevalence. Non-flagged recordings have not all been manually reviewed."
    )


if __name__ == "__main__":
    main()
