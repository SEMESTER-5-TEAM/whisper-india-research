import pandas as pd
import re
from pathlib import Path


INPUT_FILE = Path(
    "results/baseline/whisper_small_baseline_expanded.csv"
)

OUTPUT_FILE = Path(
    "results/baseline/hallucination_candidates_expanded.csv"
)


# ============================================================
# Load baseline results
# ============================================================

df = pd.read_csv(INPUT_FILE)

df = df[
    df["status"] == "success"
].copy()


# ============================================================
# Helper functions
# ============================================================

def tokens(text):
    """
    Basic whitespace tokenization.
    We deliberately keep this simple because the goal here
    is candidate generation, not final linguistic scoring.
    """
    return str(text).split()


def repetition_ratio(text, n=2):
    """
    Fraction of bigrams that are repeated.

    Example:
        "to to to to"
    has extremely high repetition.

    This is only a candidate signal.
    """

    t = tokens(text)

    if len(t) < n:
        return 0.0

    ngrams = [
        tuple(t[i:i+n])
        for i in range(len(t) - n + 1)
    ]

    if not ngrams:
        return 0.0

    unique = len(set(ngrams))

    return 1.0 - (unique / len(ngrams))


def longest_run(text):
    """
    Longest consecutive run of the same token.
    """

    t = tokens(text)

    if not t:
        return 0

    longest = 1
    current = 1

    for i in range(1, len(t)):

        if t[i] == t[i-1]:

            current += 1

            if current > longest:
                longest = current

        else:

            current = 1

    return longest


def token_length(text):
    return len(tokens(text))


# ============================================================
# Calculate candidate signals
# ============================================================

df["reference_tokens"] = (
    df["reference"]
    .fillna("")
    .apply(token_length)
)

df["prediction_tokens"] = (
    df["prediction"]
    .fillna("")
    .apply(token_length)
)

df["length_ratio"] = (
    (df["prediction_tokens"] + 1)
    /
    (df["reference_tokens"] + 1)
)

df["bigram_repetition"] = (
    df["prediction"]
    .fillna("")
    .apply(repetition_ratio)
)

df["longest_repeated_run"] = (
    df["prediction"]
    .fillna("")
    .apply(longest_run)
)


# ============================================================
# Candidate rule
# ============================================================
#
# These are deliberately conservative candidate rules.
#
# A sample becomes a candidate if:
#
#   1. The same token occurs consecutively >= 5 times
#
# OR
#
#   2. Prediction is >= 3x reference length AND
#      prediction contains substantial repetition
#
# OR
#
#   3. Prediction is very long relative to a short reference
#
# These are NOT hallucination labels.
# ============================================================

df["candidate_reason"] = ""

df.loc[
    df["longest_repeated_run"] >= 5,
    "candidate_reason"
] += "long_repetition;"

df.loc[
    (
        (df["length_ratio"] >= 3.0)
        &
        (df["bigram_repetition"] >= 0.50)
    ),
    "candidate_reason"
] += "length_plus_repetition;"

df.loc[
    (
        (df["reference_tokens"] <= 5)
        &
        (df["prediction_tokens"] >= 30)
    ),
    "candidate_reason"
] += "short_reference_long_prediction;"


candidates = df[
    df["candidate_reason"] != ""
].copy()


# ============================================================
# Sort strongest candidates first
# ============================================================

candidates["candidate_score"] = (
    candidates["longest_repeated_run"]
    +
    candidates["bigram_repetition"] * 20
    +
    candidates["length_ratio"]
)

candidates = candidates.sort_values(
    "candidate_score",
    ascending=False
)


# ============================================================
# Save
# ============================================================

candidates.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# ============================================================
# Report
# ============================================================

print("=" * 60)
print("HALLUCINATION CANDIDATE ANALYSIS")
print("=" * 60)

print(
    f"Total baseline samples: {len(df)}"
)

print(
    f"Candidate samples: {len(candidates)}"
)

print(
    f"Candidate percentage: "
    f"{100 * len(candidates) / len(df):.2f}%"
)

print("\nCandidates by dataset:")

print(
    candidates["dataset"].value_counts()
)

print("\nTop 20 candidates:")

cols = [
    "eval_id",
    "dataset",
    "state",
    "detected_language",
    "audio_duration",
    "reference",
    "prediction",
    "reference_tokens",
    "prediction_tokens",
    "length_ratio",
    "bigram_repetition",
    "longest_repeated_run",
    "candidate_reason"
]

print(
    candidates[cols]
    .head(20)
    .to_string(index=False)
)

print("\nSaved to:")

print(
    OUTPUT_FILE.resolve()
)