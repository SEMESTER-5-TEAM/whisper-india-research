
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf


ROOT = Path(__file__).resolve().parent.parent

EVAL_FILE = ROOT / "results" / "baseline" / "whisper_small_baseline_expanded.csv"
OUTPUT_FILE = ROOT / "results" / "baseline" / "acoustic_features.csv"
SUMMARY_FILE = ROOT / "results" / "baseline" / "acoustic_summary_by_dataset.csv"

DATA_DIRS = {
    "lahaja": ROOT / "data" / "raw" / "lahaja" / "data",
    "svarah": ROOT / "data" / "raw" / "svarah" / "data",
}

FRAME_MS = 20
BATCH_SIZE = 16


def frame_features(audio, sample_rate):
    """Calculate frame-level energy and zero-crossing rate."""
    frame_len = max(1, int(sample_rate * FRAME_MS / 1000))
    n_frames = len(audio) // frame_len

    if n_frames == 0:
        return None

    frames = audio[:n_frames * frame_len].reshape(n_frames, frame_len)

    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    dbfs = 20 * np.log10(np.maximum(rms, 1e-10))

    signs = frames >= 0
    zcr = np.mean(signs[:, 1:] != signs[:, :-1], axis=1)

    # Fixed thresholds make results comparable across recordings.
    silence_40 = dbfs < -40.0
    low_energy_35 = dbfs < -35.0

    return {
        "frame_count": int(n_frames),
        "silence_ratio_dbfs_40": float(np.mean(silence_40)),
        "low_energy_ratio_dbfs_35": float(np.mean(low_energy_35)),
        "mean_frame_rms": float(np.mean(rms)),
        "mean_frame_rms_dbfs": float(np.mean(dbfs)),
        "mean_zero_crossing_rate": float(np.mean(zcr)),
    }


def extract_one(wav_bytes, metadata_duration):
    """Read embedded WAV bytes and compute recording-level features."""
    with sf.SoundFile(io.BytesIO(wav_bytes)) as f:
        sample_rate = f.samplerate
        channels = f.channels
        audio = f.read(dtype="float32", always_2d=True)

    if audio.size == 0:
        raise ValueError("Empty audio")

    # Convert multiple channels to mono if necessary.
    mono = np.mean(audio, axis=1)
    duration = len(mono) / sample_rate

    features = frame_features(mono, sample_rate)
    if features is None:
        raise ValueError("Recording too short for frame analysis")

    return {
        "audio_duration_seconds": duration,
        "metadata_duration_seconds": metadata_duration,
        "duration_difference_seconds": (
            duration - metadata_duration
            if pd.notna(metadata_duration) else np.nan
        ),
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "overall_rms": float(np.sqrt(np.mean(mono ** 2))),
        **features,
    }


def main():
    eval_df = pd.read_csv(EVAL_FILE)

    required = {
        "eval_id", "dataset", "source_shard", "source_row",
        "reference", "prediction", "status"
    }
    missing = required - set(eval_df.columns)
    if missing:
        raise ValueError(f"Missing evaluation columns: {sorted(missing)}")

    # Process only successful evaluations with valid shard locations.
    eval_df = eval_df[
        (eval_df["status"] == "success")
        & eval_df["source_shard"].notna()
        & eval_df["source_row"].notna()
    ].copy()

    eval_df["source_row"] = eval_df["source_row"].astype(int)

    results = []
    errors = []

    groups = eval_df.groupby(
        ["dataset", "source_shard"], sort=False
    )

    for (dataset, shard), group in groups:
        if dataset not in DATA_DIRS:
            print(f"Skipping unknown dataset: {dataset}")
            continue

        parquet_path = DATA_DIRS[dataset] / shard
        if not parquet_path.exists():
            errors.append({
                "dataset": dataset,
                "source_shard": shard,
                "source_row": None,
                "error": "Parquet shard not found",
            })
            continue

        wanted = {
            int(row): metadata
            for row, metadata in zip(
                group["source_row"], group.to_dict("records")
            )
        }

        print(
            f"\nProcessing {dataset}/{shard}: "
            f"{len(wanted)} evaluation recordings"
        )

        parquet_file = pq.ParquetFile(parquet_path)
        row_offset = 0

        # Small batches avoid loading a whole shard's audio into memory.
        for batch in parquet_file.iter_batches(
            columns=["audio_filepath", "duration"],
            batch_size=BATCH_SIZE,
        ):
            rows = batch.to_pylist()

            for local_idx, item in enumerate(rows):
                source_row = row_offset + local_idx
                if source_row not in wanted:
                    continue

                metadata = wanted[source_row]
                try:
                    audio_field = item.get("audio_filepath") or {}
                    wav_bytes = audio_field.get("bytes")

                    if not wav_bytes:
                        raise ValueError(
                            "No embedded audio bytes in audio_filepath"
                        )

                    features = extract_one(
                        wav_bytes, item.get("duration")
                    )

                    results.append({
                        **metadata,
                        **features,
                        "acoustic_status": "success",
                        "acoustic_error": "",
                    })

                except Exception as exc:
                    errors.append({
                        "dataset": dataset,
                        "source_shard": shard,
                        "source_row": source_row,
                        "error": str(exc),
                    })

                if len(results) and len(results) % 100 == 0:
                    print(f"  Extracted {len(results)} recordings")

            row_offset += len(rows)

    if not results:
        raise RuntimeError(
            "No acoustic features extracted. Check shard mapping and errors."
        )

    out = pd.DataFrame(results)

    # Compute transcript error metrics without changing existing results.
    def word_error_rate(row):
        ref = str(row["reference"] or "").split()
        pred = str(row["prediction"] or "").split()
        if not ref:
            return np.nan

        # Levenshtein distance using a memory-efficient dynamic program.
        previous = list(range(len(pred) + 1))
        for i, word in enumerate(ref, start=1):
            current = [i]
            for j, predicted_word in enumerate(pred, start=1):
                current.append(min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (word != predicted_word),
                ))
            previous = current

        return previous[-1] / len(ref)

    out["record_wer"] = out.apply(word_error_rate, axis=1)

    # Aggregate acoustic properties and WER by dataset.
    summary = out.groupby("dataset").agg(
        recordings=("eval_id", "count"),
        mean_duration_seconds=("audio_duration_seconds", "mean"),
        median_duration_seconds=("audio_duration_seconds", "median"),
        mean_silence_ratio_dbfs_40=("silence_ratio_dbfs_40", "mean"),
        mean_low_energy_ratio_dbfs_35=("low_energy_ratio_dbfs_35", "mean"),
        mean_rms=("overall_rms", "mean"),
        mean_zero_crossing_rate=("mean_zero_crossing_rate", "mean"),
        mean_record_wer=("record_wer", "mean"),
    ).reset_index()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)

    if errors:
        pd.DataFrame(errors).to_csv(
            OUTPUT_FILE.with_name("acoustic_extraction_errors.csv"),
            index=False,
        )

    print("\nExtraction complete.")
    print(f"Feature rows: {len(out)}")
    print(f"Errors: {len(errors)}")
    print(f"Feature file: {OUTPUT_FILE}")
    print(f"Dataset summary: {SUMMARY_FILE}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
