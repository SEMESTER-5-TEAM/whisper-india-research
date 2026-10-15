

from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results" / "baseline"
FORCED_DIR = ROOT / "results" / "forced_language"
META = ROOT / "data" / "metadata" / "evaluation_subset_60_per_state.csv"

BASELINE_FILE = BASE / "whisper_small_baseline_expanded.csv"
FORCED_FILE = FORCED_DIR / "whisper_small_forced_language.csv"
OUT = FORCED_DIR / "speaker_aware_validation.csv"

BOOTSTRAP_REPS = 5000
SEED = 42


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"[^\w\s]|_", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


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


def get_errors(reference, prediction):
    ref = normalize(reference)
    hyp = normalize(prediction)
    return (
        edit_distance(ref.split(), hyp.split()),
        len(ref.split()),
        edit_distance(ref, hyp),
        len(ref),
    )


def corpus_rate(errors, denominator, indices):
    den = denominator[indices].sum()
    return errors[indices].sum() / den if den else np.nan


def main():
    baseline = pd.read_csv(BASELINE_FILE)
    forced = pd.read_csv(FORCED_FILE)
    meta = pd.read_csv(META)

    keys = ["eval_id", "dataset", "record_id"]
    baseline = baseline[baseline["status"].eq("success")].copy()
    forced = forced[forced["status"].eq("success")].copy()

    merged = baseline.merge(
        forced, on=keys, how="inner",
        suffixes=("_base", "_forced"), validate="one_to_one"
    )

    speaker_meta = meta[["eval_id", "dataset", "record_id", "sp_id"]]
    merged = merged.merge(
        speaker_meta, on=keys, how="left", validate="one_to_one"
    )

    merged["detected_language"] = (
        merged["detected_language"].fillna("").astype(str).str.lower()
    )
    merged["expected_language"] = (
        merged["expected_language"].fillna("").astype(str).str.lower()
    )
    mismatch = (
        merged["detected_language"] != merged["expected_language"]
    )
    merged = merged[mismatch].copy()

    # Avoid silently comparing different references or empty outputs.
    rows = []
    for _, r in merged.iterrows():
        ref_a = normalize(r["reference_base"])
        ref_b = normalize(r["reference_forced"])
        if ref_a != ref_b:
            raise ValueError(f"Reference mismatch: {r['eval_id']}")

        pred_a = "" if pd.isna(r["prediction"]) else str(r["prediction"])
        pred_b = "" if pd.isna(r["forced_prediction"]) else str(r["forced_prediction"])
        if not pred_a.strip() or not pred_b.strip() or not ref_a:
            continue

        we_a, nw, ce_a, nc = get_errors(ref_a, pred_a)
        we_b, _, ce_b, _ = get_errors(ref_b, pred_b)

        rows.append({
            "eval_id": r["eval_id"],
            "dataset": r["dataset"],
            "sp_id": r["sp_id"],
            "base_word_errors": we_a,
            "forced_word_errors": we_b,
            "reference_words": nw,
            "base_char_errors": ce_a,
            "forced_char_errors": ce_b,
            "reference_chars": nc,
        })

    data = pd.DataFrame(rows)
    rng = np.random.default_rng(SEED)
    output = []

    for dataset, part in data.groupby("dataset"):
        part = part.reset_index(drop=True)
        cluster_mode = (
            "speaker" if dataset == "lahaja"
            and part["sp_id"].notna().all()
            else "recording"
        )

        if cluster_mode == "speaker":
            clusters = part["sp_id"].astype(str).to_numpy()
            unique_clusters = np.unique(clusters)
            cluster_indices = {
                c: np.flatnonzero(clusters == c)
                for c in unique_clusters
            }
            units = [
                cluster_indices[c] for c in unique_clusters
            ]
        else:
            units = [np.array([i]) for i in range(len(part))]

        print(
            f"\n{dataset.upper()}: n={len(part)}, "
            f"bootstrap unit={cluster_mode}, units={len(units)}"
        )

        for metric, base_col, forced_col, den_col in [
            ("WER", "base_word_errors", "forced_word_errors",
             "reference_words"),
            ("CER", "base_char_errors", "forced_char_errors",
             "reference_chars"),
        ]:
            base_err = part[base_col].to_numpy(float)
            forced_err = part[forced_col].to_numpy(float)
            denom = part[den_col].to_numpy(float)
            idx = np.arange(len(part))

            base_rate = corpus_rate(base_err, denom, idx)
            forced_rate = corpus_rate(forced_err, denom, idx)
            delta = forced_rate - base_rate

            boot = np.empty(BOOTSTRAP_REPS)
            for b in range(BOOTSTRAP_REPS):
                sampled_units = rng.integers(0, len(units), len(units))
                sampled_idx = np.concatenate(
                    [units[j] for j in sampled_units]
                )
                boot[b] = (
                    corpus_rate(forced_err, denom, sampled_idx)
                    - corpus_rate(base_err, denom, sampled_idx)
                )

            lo, hi = np.quantile(boot, [0.025, 0.975])
            output.append({
                "dataset": dataset,
                "metric": metric,
                "recordings": len(part),
                "bootstrap_unit": cluster_mode,
                "units": len(units),
                "baseline_corpus_rate": base_rate,
                "forced_corpus_rate": forced_rate,
                "delta_percentage_points": 100 * delta,
                "ci_low_percentage_points": 100 * lo,
                "ci_high_percentage_points": 100 * hi,
            })

            print(
                f"{metric}: baseline={base_rate:.4f}, "
                f"forced={forced_rate:.4f}, "
                f"change={100*delta:+.2f} pp, "
                f"95% CI=[{100*lo:+.2f}, {100*hi:+.2f}] pp"
            )

    result = pd.DataFrame(output)
    result.to_csv(OUT, index=False)
    print("\nSaved:", OUT)


if __name__ == "__main__":
    main()
