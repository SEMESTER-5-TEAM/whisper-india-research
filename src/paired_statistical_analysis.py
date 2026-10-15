
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BASELINE_PATH = Path(
    "results/baseline/whisper_small_baseline_expanded.csv"
)
INTERVENTION_PATH = Path(
    "results/intervention/whisper_small_no_previous_context.csv"
)
OUTPUT_PATH = Path(
    "results/intervention/paired_statistical_analysis.csv"
)

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


def paired_wilcoxon(baseline, intervention):
    differences = intervention - baseline

    if np.allclose(differences, 0):
        return 1.0

    try:
        return float(
            wilcoxon(
                baseline,
                intervention,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            ).pvalue
        )
    except ValueError:
        return np.nan


def holm_adjust(p_values):
    """Holm correction for multiple hypothesis tests."""
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


def main():
    baseline = pd.read_csv(BASELINE_PATH)
    intervention = pd.read_csv(INTERVENTION_PATH)

    # Only compare successful predictions.
    baseline = baseline[baseline["status"].eq("success")].copy()
    intervention = intervention[
        intervention["status"].eq("success")
    ].copy()

    keys = ["eval_id", "dataset", "record_id"]

    if baseline.duplicated(keys).any():
        raise ValueError("Duplicate recording keys in baseline.")
    if intervention.duplicated(keys).any():
        raise ValueError("Duplicate recording keys in intervention.")

    merged = baseline.merge(
        intervention,
        on=keys,
        how="inner",
        suffixes=("_base", "_inter"),
        validate="one_to_one",
    )

    print("Baseline successful rows:", len(baseline))
    print("Intervention successful rows:", len(intervention))
    print("Matched recordings:", len(merged))

    if len(merged) == 0:
        raise ValueError("No matched recordings found.")

    # Ensure references are identical after the existing normalization.
    ref_base = merged["reference_base"].fillna("").map(normalize)
    ref_inter = merged["reference_inter"].fillna("").map(normalize)

    if not ref_base.equals(ref_inter):
        raise ValueError(
            "Reference transcripts differ between conditions."
        )

    records = []

    for _, row in merged.iterrows():
        reference = normalize(row["reference_base"])
        pred_base = normalize(row["prediction_base"])
        pred_inter = normalize(row["prediction_inter"])

        ref_words = reference.split()
        base_words = pred_base.split()
        inter_words = pred_inter.split()

        ref_chars = len(reference)

        records.append({
            "eval_id": row["eval_id"],
            "dataset": row["dataset"],
            "record_id": row["record_id"],
            "state": row.get("state_base", ""),
            "reference_words": len(ref_words),
            "reference_chars": ref_chars,
            "base_word_errors": edit_distance(ref_words, base_words),
            "inter_word_errors": edit_distance(ref_words, inter_words),
            "base_char_errors": edit_distance(reference, pred_base),
            "inter_char_errors": edit_distance(reference, pred_inter),
        })

    data = pd.DataFrame(records)
    rng = np.random.default_rng(RANDOM_SEED)
    results = []

    for dataset, group in data.groupby("dataset"):
        group = group.reset_index(drop=True)
        n = len(group)

        print(f"\nDataset: {dataset} | matched recordings: {n}")

        metric_specs = [
            (
                "WER",
                "base_word_errors",
                "inter_word_errors",
                "reference_words",
            ),
            (
                "CER",
                "base_char_errors",
                "inter_char_errors",
                "reference_chars",
            ),
        ]

        for metric, base_col, inter_col, denom_col in metric_specs:
            base_errors = group[base_col].to_numpy(dtype=float)
            inter_errors = group[inter_col].to_numpy(dtype=float)
            denominators = group[denom_col].to_numpy(dtype=float)

            valid = denominators > 0
            base_errors = base_errors[valid]
            inter_errors = inter_errors[valid]
            denominators = denominators[valid]

            n_valid = len(denominators)
            if n_valid == 0:
                continue

            indices_all = np.arange(n_valid)
            base_rate = corpus_rate(
                base_errors, denominators, indices_all
            )
            inter_rate = corpus_rate(
                inter_errors, denominators, indices_all
            )
            delta = inter_rate - base_rate

            # Paired bootstrap: resample recordings and retain both
            # conditions together for every sampled recording.
            bootstrap_deltas = np.empty(BOOTSTRAP_REPS)

            for b in range(BOOTSTRAP_REPS):
                sample = rng.integers(0, n_valid, size=n_valid)
                bootstrap_deltas[b] = (
                    corpus_rate(inter_errors, denominators, sample)
                    - corpus_rate(base_errors, denominators, sample)
                )

            ci_low, ci_high = np.quantile(
                bootstrap_deltas, [0.025, 0.975]
            )

            # Per-recording rates for the paired signed-rank test.
            base_per_record = base_errors / denominators
            inter_per_record = inter_errors / denominators

            p_value = paired_wilcoxon(
                base_per_record, inter_per_record
            )

            result = {
                "dataset": dataset,
                "metric": metric,
                "recordings": n_valid,
                "baseline_corpus_rate": base_rate,
                "intervention_corpus_rate": inter_rate,
                "delta_intervention_minus_baseline": delta,
                "delta_percentage_points": 100 * delta,
                "bootstrap_95ci_low": ci_low,
                "bootstrap_95ci_high": ci_high,
                "bootstrap_95ci_low_pp": 100 * ci_low,
                "bootstrap_95ci_high_pp": 100 * ci_high,
                "wilcoxon_p": p_value,
            }
            results.append(result)

            print(
                f"{metric}: baseline={base_rate:.4f}, "
                f"intervention={inter_rate:.4f}, "
                f"delta={delta:+.4f} "
                f"({100 * delta:+.2f} percentage points)"
            )
            print(
                f"  Bootstrap 95% CI: "
                f"[{ci_low:+.4f}, {ci_high:+.4f}]"
            )
            print(f"  Paired Wilcoxon p-value: {p_value:.6g}")

    output = pd.DataFrame(results)

    # Correct the four dataset-by-metric p-values together.
    output["wilcoxon_p_holm"] = holm_adjust(
        output["wilcoxon_p"].to_numpy()
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    print("\nFINAL RESULTS")
    print(output.round(5).to_string(index=False))
    print("\nSaved:", OUTPUT_PATH)
    print(
        "\nInterpretation: negative delta means lower corpus error "
        "for the intervention. The bootstrap resamples recordings; "
        "it does not account for speaker clustering."
    )


if __name__ == "__main__":
    main()
