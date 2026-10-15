import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BASELINE_PATH = Path(
    "results/baseline/whisper_small_baseline_expanded.csv"
)
FORCED_PATH = Path(
    "results/forced_language/whisper_small_forced_language.csv"
)
OUTPUT_DIR = Path("results/forced_language")
OUTPUT_METRICS = OUTPUT_DIR / "paired_metrics.csv"
OUTPUT_CANDIDATES = OUTPUT_DIR / "candidate_comparison.csv"

BOOTSTRAP_REPS = 2000
RANDOM_SEED = 42


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"[^\w\s]|_", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))

    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (item_a != item_b),
                )
            )
        previous = current

    return previous[-1]


def corpus_rate(errors, denominators, indices):
    denominator = denominators[indices].sum()
    if denominator == 0:
        return np.nan
    return errors[indices].sum() / denominator


def paired_wilcoxon(baseline, forced):
    differences = forced - baseline

    if np.allclose(differences, 0):
        return 1.0

    try:
        return float(
            wilcoxon(
                baseline,
                forced,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            ).pvalue
        )
    except ValueError:
        return np.nan


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan)
    valid = np.where(np.isfinite(p_values))[0]

    if len(valid) == 0:
        return adjusted

    order = valid[np.argsort(p_values[valid])]
    m = len(order)
    running_max = 0.0

    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p_values[index])
        running_max = max(running_max, value)
        adjusted[index] = running_max

    return adjusted


def hallucination_candidate(reference, prediction):
    """
    Apply the original project's screening rules to RAW text.

    This intentionally uses whitespace tokenization only, matching
    src/find_hallucination_candidates_expanded.py. Do not normalize
    punctuation or letter case for candidate detection.
    """
    reference = "" if pd.isna(reference) else str(reference)
    prediction = "" if pd.isna(prediction) else str(prediction)

    ref_tokens = reference.split()
    pred_tokens = prediction.split()

    # A run of at least five identical consecutive tokens.
    longest_run = 0
    current_run = 0
    previous_token = None

    for token in pred_tokens:
        if token == previous_token:
            current_run += 1
        else:
            current_run = 1
            previous_token = token
        longest_run = max(longest_run, current_run)

    if longest_run >= 5:
        return True

    length_ratio = (len(pred_tokens) + 1) / (len(ref_tokens) + 1)

    # Fraction of adjacent prediction bigrams that are repeated.
    bigrams = list(zip(pred_tokens, pred_tokens[1:]))

    if bigrams:
        repeated_bigrams = len(bigrams) - len(set(bigrams))
        repetition_ratio = repeated_bigrams / len(bigrams)
    else:
        repetition_ratio = 0.0

    if length_ratio >= 3 and repetition_ratio >= 0.50:
        return True

    if len(ref_tokens) <= 5 and len(pred_tokens) >= 30:
        return True

    return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline = pd.read_csv(BASELINE_PATH)
    forced = pd.read_csv(FORCED_PATH)

    baseline = baseline[baseline["status"].eq("success")].copy()
    forced = forced[forced["status"].eq("success")].copy()

    keys = ["eval_id", "dataset", "record_id"]

    if baseline.duplicated(keys).any():
        raise ValueError("Duplicate keys in baseline.")
    if forced.duplicated(keys).any():
        raise ValueError("Duplicate keys in forced-language results.")

    merged = baseline.merge(
        forced,
        on=keys,
        how="inner",
        suffixes=("_base", "_forced"),
        validate="one_to_one",
    )

    print("Successful baseline rows:", len(baseline))
    print("Successful forced-language rows:", len(forced))
    print("Matched recordings:", len(merged))

    if merged.empty:
        raise ValueError("No matched recordings found.")

    # Confirm the same references were used in both conditions.
    ref_base = merged["reference_base"].fillna("").map(normalize)
    ref_forced = merged["reference_forced"].fillna("").map(normalize)

    if not ref_base.equals(ref_forced):
        raise ValueError("Reference transcripts differ between conditions.")

    # Ensure this is the intended language-mismatch subset.
    mismatch = (
    merged["detected_language"].fillna("").str.lower()
    != merged["expected_language"].fillna("").str.lower()
    )
    if not mismatch.all():
        raise ValueError(
            "Some matched rows do not meet the language-mismatch rule."
        )

    records = []

    for _, row in merged.iterrows():
        # Normalized text is used for WER/CER calculations.
        reference = normalize(row["reference_base"])
        pred_base = normalize(row["prediction"])
        pred_forced = normalize(row["forced_prediction"])

        # Raw text is used for candidate detection to match the original
        # detector in src/find_hallucination_candidates_expanded.py.
        raw_reference = row["reference_base"]
        raw_pred_base = row["prediction"]
        raw_pred_forced = row["forced_prediction"]

        ref_words = reference.split()

        records.append({
            "eval_id": row["eval_id"],
            "dataset": row["dataset"],
            "record_id": row["record_id"],
            "reference_words": len(ref_words),
            "reference_chars": len(reference),
            "base_word_errors": edit_distance(
                ref_words, pred_base.split()
            ),
            "forced_word_errors": edit_distance(
                ref_words, pred_forced.split()
            ),
            "base_char_errors": edit_distance(reference, pred_base),
            "forced_char_errors": edit_distance(reference, pred_forced),
            "base_candidate": hallucination_candidate(
                raw_reference, raw_pred_base
            ),
            "forced_candidate": hallucination_candidate(
                raw_reference, raw_pred_forced
            ),
        })

    data = pd.DataFrame(records)
    rng = np.random.default_rng(RANDOM_SEED)
    metric_results = []
    candidate_results = []

    for dataset, group in data.groupby("dataset"):
        group = group.reset_index(drop=True)
        print(f"\nDataset: {dataset} | n={len(group)}")

        for metric, base_col, forced_col, denom_col in [
            (
                "WER",
                "base_word_errors",
                "forced_word_errors",
                "reference_words",
            ),
            (
                "CER",
                "base_char_errors",
                "forced_char_errors",
                "reference_chars",
            ),
        ]:
            valid = group[denom_col].to_numpy(dtype=float) > 0
            subset = group.loc[valid]

            base_errors = subset[base_col].to_numpy(dtype=float)
            forced_errors = subset[forced_col].to_numpy(dtype=float)
            denominators = subset[denom_col].to_numpy(dtype=float)
            n = len(subset)

            indices = np.arange(n)
            base_rate = corpus_rate(
                base_errors, denominators, indices
            )
            forced_rate = corpus_rate(
                forced_errors, denominators, indices
            )
            delta = forced_rate - base_rate

            bootstrap_deltas = np.empty(BOOTSTRAP_REPS)

            for b in range(BOOTSTRAP_REPS):
                sample = rng.integers(0, n, size=n)
                bootstrap_deltas[b] = (
                    corpus_rate(
                        forced_errors, denominators, sample
                    )
                    - corpus_rate(
                        base_errors, denominators, sample
                    )
                )

            ci_low, ci_high = np.quantile(
                bootstrap_deltas, [0.025, 0.975]
            )

            base_per_record = base_errors / denominators
            forced_per_record = forced_errors / denominators

            p_value = paired_wilcoxon(
                base_per_record, forced_per_record
            )

            metric_results.append({
                "dataset": dataset,
                "metric": metric,
                "recordings": n,
                "baseline_corpus_rate": base_rate,
                "forced_corpus_rate": forced_rate,
                "delta_forced_minus_baseline": delta,
                "delta_percentage_points": 100 * delta,
                "bootstrap_95ci_low": ci_low,
                "bootstrap_95ci_high": ci_high,
                "bootstrap_95ci_low_pp": 100 * ci_low,
                "bootstrap_95ci_high_pp": 100 * ci_high,
                "wilcoxon_p": p_value,
            })

            print(
                f"{metric}: baseline={base_rate:.4f}, "
                f"forced={forced_rate:.4f}, "
                f"delta={100 * delta:+.2f} percentage points"
            )
            print(
                f"  Recording-bootstrap 95% CI: "
                f"[{100 * ci_low:+.2f}, {100 * ci_high:+.2f}] pp"
            )
            print(f"  Wilcoxon p={p_value:.6g}")

        # Candidate flags are a screening measure, not verified
        # hallucination labels. Compare them descriptively.
        base_flags = group["base_candidate"].astype(bool)
        forced_flags = group["forced_candidate"].astype(bool)

        neither = int((~base_flags & ~forced_flags).sum())
        both = int((base_flags & forced_flags).sum())
        base_only = int((base_flags & ~forced_flags).sum())
        forced_only = int((~base_flags & forced_flags).sum())

        candidate_results.append({
            "dataset": dataset,
            "recordings": len(group),
            "baseline_candidates": int(base_flags.sum()),
            "baseline_candidate_rate": float(base_flags.mean()),
            "forced_candidates": int(forced_flags.sum()),
            "forced_candidate_rate": float(forced_flags.mean()),
            "both_conditions_flagged": both,
            "baseline_only_flagged": base_only,
            "forced_only_flagged": forced_only,
            "neither_flagged": neither,
        })

        print(
            f"Candidate flags: baseline={base_flags.sum()}/"
            f"{len(group)}, forced={forced_flags.sum()}/{len(group)}"
        )
        print(
            f"  Both={both}, baseline-only={base_only}, "
            f"forced-only={forced_only}, neither={neither}"
        )

    metrics = pd.DataFrame(metric_results)
    metrics["wilcoxon_p_holm"] = holm_adjust(
        metrics["wilcoxon_p"].to_numpy()
    )
    metrics.to_csv(OUTPUT_METRICS, index=False)

    pd.DataFrame(candidate_results).to_csv(
        OUTPUT_CANDIDATES, index=False
    )

    print("\nFINAL METRIC RESULTS")
    print(metrics.round(5).to_string(index=False))

    print("\nCANDIDATE COMPARISON")
    print(pd.DataFrame(candidate_results).to_string(index=False))

    print("\nSaved:", OUTPUT_METRICS)
    print("Saved:", OUTPUT_CANDIDATES)
    print(
        "\nInterpretation: negative WER/CER delta means lower error "
        "after language forcing. Confidence intervals resample "
        "recordings, not speakers. Candidate flags are not ground-truth "
        "hallucination labels."
    )


if __name__ == "__main__":
    main()
