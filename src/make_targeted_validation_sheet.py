#!/usr/bin/env python
"""Create a targeted, blinded review sheet for the new four-condition experiment.

Run from project root:
    python src\make_targeted_validation_sheet.py

Selects up to 30 unique recordings where the automatic error category differs
between the four conditions. Review all four outputs against the reference/audio.
The reviewer sheet omits automatic labels; a separate key preserves them.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "lora_generic_hindi"
INPUT = RESULTS / "four_condition_error_class_by_record.csv"
SEED = 2026
TARGET_N = 30
CONDITIONS = [
    "baseline_forced_hindi", "lora_forced_hindi",
    "baseline_default", "lora_default",
]


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing {INPUT}. Run src\\score_four_conditions.py and "
            "src\\classify_errors.py first."
        )
    df = pd.read_csv(INPUT, keep_default_na=False)
    required = {
        "condition", "source_shard", "source_row", "sp_id", "reference",
        "prediction", "error_class", "status"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input missing columns: {sorted(missing)}")

    keys = ["source_shard", "source_row"]
    base = df[df["condition"] == CONDITIONS[0]].copy()
    if base.duplicated(keys).any():
        raise ValueError("Duplicate recording keys in the baseline condition")

    wide = base[keys + [c for c in ["sp_id", "state", "district", "reference"] if c in base.columns]].copy()
    class_columns = []
    for condition in CONDITIONS:
        part = df[df["condition"] == condition][keys + ["prediction", "error_class", "status"]].copy()
        if part.duplicated(keys).any():
            raise ValueError(f"Duplicate recording keys in {condition}")
        part = part.rename(columns={
            "prediction": f"prediction__{condition}",
            "error_class": f"class__{condition}",
            "status": f"status__{condition}",
        })
        class_columns.append(f"class__{condition}")
        wide = wide.merge(part, on=keys, how="left", validate="one_to_one")
        if wide[f"prediction__{condition}"].isna().any():
            raise ValueError(f"Missing predictions for {condition}")

    # Ensure the same set of recordings is represented in every condition.
    if len(wide) != len(base):
        raise ValueError("Recording alignment failed")
    wide["number_of_distinct_classes"] = wide[class_columns].nunique(axis=1)
    # Prioritize class disagreement, then a deterministic random tie-break.
    rng = np.random.default_rng(SEED)
    wide["_random_tiebreak"] = rng.random(len(wide))
    candidates = wide[wide["number_of_distinct_classes"] > 1].copy()
    candidates = candidates.sort_values(
        ["number_of_distinct_classes", "_random_tiebreak"],
        ascending=[False, True]
    )
    selected = candidates.head(TARGET_N).copy()
    if len(selected) < TARGET_N:
        used = set(map(tuple, selected[keys].to_numpy()))
        remaining = wide[~wide[keys].apply(tuple, axis=1).isin(used)]
        remaining = remaining.sort_values("_random_tiebreak")
        selected = pd.concat([selected, remaining.head(TARGET_N-len(selected))], ignore_index=True)

    selected = selected.iloc[rng.permutation(len(selected))].reset_index(drop=True)
    selected.insert(0, "review_id", [f"TV{i:03d}" for i in range(1, len(selected)+1)])

    reviewer_columns = ["review_id"] + keys
    for col in ["sp_id", "state", "district", "reference"]:
        if col in selected.columns:
            reviewer_columns.append(col)
    reviewer_columns += [f"prediction__{c}" for c in CONDITIONS]
    review = selected[reviewer_columns].copy()
    review["audio_checked"] = ""
    for condition in CONDITIONS:
        review[f"human_label__{condition}"] = ""
    review["notes"] = ""
    review["label_guide"] = (
        "Label each output independently after listening: VALID / ORDINARY_ASR_ERROR / "
        "HALLUCINATION / WRONG_LANGUAGE / UNCERTAIN. A script mismatch or repetition "
        "alone does not prove hallucination."
    )

    key_columns = ["review_id"] + keys + ["number_of_distinct_classes"] + class_columns
    key = selected[key_columns].copy()
    review_path = RESULTS / "targeted_human_validation_sheet.csv"
    key_path = RESULTS / "targeted_human_validation_key.csv"
    review.to_csv(review_path, index=False)
    key.to_csv(key_path, index=False)

    print(f"Selected recordings: {len(review)}")
    print(f"Candidates with class disagreement available: {len(candidates)}")
    print(f"Saved reviewer sheet: {review_path.relative_to(ROOT)}")
    print(f"Saved separate class key: {key_path.relative_to(ROOT)}")
    print("The reviewer sheet does not include automatic class labels.")
    print("Review all four outputs for each recording using the corresponding audio.")
    if len(candidates) < TARGET_N:
        print("Note: fewer than 30 recordings had class disagreement; remaining rows were sampled from other recordings.")


if __name__ == "__main__":
    main()
