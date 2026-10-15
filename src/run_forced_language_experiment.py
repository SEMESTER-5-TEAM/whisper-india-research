import io
import time
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import whisper

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

MODEL_NAME = "small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

METADATA_FILE = Path(
    "data/metadata/evaluation_subset_60_per_state.csv"
)

BASELINE_FILE = Path(
    "results/baseline/whisper_small_baseline_expanded.csv"
)

OUTPUT_DIR = Path("results/forced_language")
OUTPUT_FILE = OUTPUT_DIR / "whisper_small_forced_language.csv"

CHECKPOINT_EVERY = 10

EXPECTED_LANGUAGE = {
    "lahaja": "hi",
    "svarah": "en",
}


def decode_audio(audio_filepath):
    """Decode audio embedded in a Parquet audio object."""
    if not isinstance(audio_filepath, dict):
        raise TypeError(
            f"Expected audio dictionary, got {type(audio_filepath)}"
        )

    audio, sample_rate = sf.read(
        io.BytesIO(audio_filepath["bytes"]),
        dtype="float32",
    )
    return audio, sample_rate


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading metadata and baseline results...")

    metadata = pd.read_csv(METADATA_FILE)
    baseline = pd.read_csv(BASELINE_FILE)

    # Keep the original baseline reference and predictions.
    required_columns = [
        "eval_id",
        "dataset",
        "record_id",
        "detected_language",
        "reference",
        "prediction",
        "status",
    ]

    missing = [
        column for column in required_columns
        if column not in baseline.columns
    ]
    if missing:
        raise ValueError(f"Missing baseline columns: {missing}")

    baseline = baseline[required_columns].copy()
    baseline["eval_id"] = baseline["eval_id"].astype(int)
    metadata["eval_id"] = metadata["eval_id"].astype(int)

    baseline = baseline[baseline["status"] == "success"].copy()

    baseline["expected_language"] = (
        baseline["dataset"].map(EXPECTED_LANGUAGE)
    )

    # Target only recordings where detected and expected
    # language differ. This is a screening rule, not proof
    # that the automatic language detection was wrong.
    targets = baseline[
        baseline["expected_language"].notna()
        & (
            baseline["detected_language"].str.lower()
            != baseline["expected_language"]
        )
    ].copy()

    targets = targets.merge(
        metadata,
        on=["eval_id", "dataset", "record_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_meta"),
    )

    if targets["source_shard"].isna().any():
        missing_rows = int(targets["source_shard"].isna().sum())
        raise ValueError(
            f"{missing_rows} target rows have no source shard."
        )

    # Resume support: preserve previously completed output rows.
    if OUTPUT_FILE.exists():
        results = pd.read_csv(OUTPUT_FILE)
        completed_ids = set(results["eval_id"].astype(int))
        print(f"Existing output rows: {len(results)}")
    else:
        results = pd.DataFrame()
        completed_ids = set()

    pending = targets[
        ~targets["eval_id"].isin(completed_ids)
    ].copy()

    print(f"Baseline successful recordings: {len(baseline)}")
    print(f"Target recordings: {len(targets)}")
    print(f"Already completed: {len(completed_ids)}")
    print(f"Remaining: {len(pending)}")

    if pending.empty:
        print("Nothing to process. Output is already complete.")
        return

    print(f"Loading Whisper {MODEL_NAME} on {DEVICE}...")
    model = whisper.load_model(MODEL_NAME, device=DEVICE)
    print("Model loaded.")

    # Cache each Parquet shard so it is read only once
    # while processing consecutive recordings from that shard.
    shard_cache = {}
    new_results = []
    start_time = time.time()

    for index, (_, row) in enumerate(pending.iterrows(), start=1):
        dataset = row["dataset"]
        shard_name = row["source_shard"]
        source_row = int(row["source_row"])
        expected_language = row["expected_language"]

        try:
            if dataset == "lahaja":
                shard_path = Path("data/raw/lahaja/data") / shard_name
            elif dataset == "svarah":
                shard_path = Path("data/raw/svarah/data") / shard_name
            else:
                raise ValueError(f"Unexpected dataset: {dataset}")

            cache_key = str(shard_path)

            if cache_key not in shard_cache:
                print(f"Reading shard: {shard_path.name}")
                shard_cache[cache_key] = pd.read_parquet(shard_path)

            shard = shard_cache[cache_key]

            if source_row < 0 or source_row >= len(shard):
                raise IndexError(
                    f"source_row {source_row} outside shard bounds"
                )

            audio_object = shard.iloc[source_row]["audio_filepath"]
            audio, sample_rate = decode_audio(audio_object)

            # Whisper's transcribe() accepts the decoded audio array.
            # Language is the only intervention; temperature and
            # FP16 settings match the baseline.
            result = model.transcribe(
                audio,
                language=expected_language,
                fp16=False,
                temperature=0,
            )

            output_row = {
                "eval_id": int(row["eval_id"]),
                "dataset": dataset,
                "record_id": row["record_id"],
                "state": row.get("state", row.get("native_state", "")),
                "district": row.get(
                    "district", row.get("native_district", "")
                ),
                "baseline_detected_language": row["detected_language"],
                "expected_language": expected_language,
                "reference": row["reference"],
                "baseline_prediction": row["prediction"],
                "forced_prediction": result.get("text", "").strip(),
                "forced_detected_language": result.get("language", ""),
                "audio_duration": len(audio) / sample_rate,
                "sample_rate": sample_rate,
                "processing_seconds": None,
                "status": "success",
                "error": "",
            }

        except Exception as exc:
            output_row = {
                "eval_id": int(row["eval_id"]),
                "dataset": dataset,
                "record_id": row["record_id"],
                "state": row.get("state", row.get("native_state", "")),
                "district": row.get(
                    "district", row.get("native_district", "")
                ),
                "baseline_detected_language": row["detected_language"],
                "expected_language": expected_language,
                "reference": row["reference"],
                "baseline_prediction": row["prediction"],
                "forced_prediction": "",
                "forced_detected_language": "",
                "audio_duration": None,
                "sample_rate": None,
                "processing_seconds": None,
                "status": "error",
                "error": repr(exc),
            }
            print(
                f"ERROR eval_id={int(row['eval_id'])}: {exc}"
            )

        new_results.append(output_row)

        if index % CHECKPOINT_EVERY == 0 or index == len(pending):
            combined = pd.concat(
                [results, pd.DataFrame(new_results)],
                ignore_index=True,
            )
            combined.to_csv(OUTPUT_FILE, index=False)
            results = combined
            new_results = []

            elapsed = time.time() - start_time
            print(
                f"Progress: {index}/{len(pending)} | "
                f"Saved: {len(results)} | "
                f"Elapsed: {elapsed / 60:.1f} min"
            )

    # Save any final rows not already checkpointed.
    if new_results:
        results = pd.concat(
            [results, pd.DataFrame(new_results)],
            ignore_index=True,
        )
        results.to_csv(OUTPUT_FILE, index=False)

    print("\nExperiment finished.")
    print(f"Results saved to: {OUTPUT_FILE}")
    print(results["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
