import io
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]

SAMPLE_FILE = (
    ROOT / "results" / "baseline"
    / "unflagged_random_sample_100.csv"
)
METADATA_FILE = (
    ROOT / "data" / "metadata"
    / "evaluation_subset_60_per_state.csv"
)
OUTPUT_DIR = (
    ROOT / "results" / "baseline"
    / "unflagged_review_audio"
)

DATA_DIRS = {
    "lahaja": ROOT / "data" / "raw" / "lahaja" / "data",
    "svarah": ROOT / "data" / "raw" / "svarah" / "data",
}


def main():
    sample = pd.read_csv(SAMPLE_FILE)
    metadata = pd.read_csv(METADATA_FILE)

    keys = ["eval_id", "dataset"]

    if sample.duplicated(keys).any():
        raise ValueError("Duplicate sample eval_id/dataset pairs.")

    metadata = metadata[keys + [
        "record_id", "source_shard", "source_row"
    ]].copy()

    if metadata.duplicated(keys).any():
        raise ValueError("Duplicate metadata eval_id/dataset pairs.")

    merged = sample.merge(
        metadata,
        on=keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    missing = merged["_merge"] != "both"
    if missing.any():
        print(merged.loc[missing, keys].to_string(index=False))
        raise ValueError(
            f"{int(missing.sum())} sampled recordings lack metadata."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = {}
    manifest = []
    failed = []

    for _, row in merged.iterrows():
        dataset = str(row["dataset"]).lower()
        shard_name = str(row["source_shard"])
        source_row = int(row["source_row"])
        eval_id = int(row["eval_id"])

        shard_path = DATA_DIRS[dataset] / shard_name
        cache_key = (dataset, shard_name)

        try:
            if cache_key not in cache:
                print(f"Loading shard: {shard_path.name}")
                cache[cache_key] = pq.read_table(
                    shard_path
                ).to_pandas()

            shard = cache[cache_key]

            if not 0 <= source_row < len(shard):
                raise IndexError(
                    f"source_row {source_row} outside shard bounds."
                )

            audio_obj = shard.iloc[source_row]["audio_filepath"]

            if not isinstance(audio_obj, dict):
                raise TypeError("Audio field is not a dictionary.")

            audio_bytes = audio_obj.get("bytes")
            if audio_bytes is None:
                raise ValueError("Embedded audio bytes are missing.")

            waveform, sample_rate = sf.read(
                io.BytesIO(audio_bytes),
                dtype="float32",
            )

            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1)

            filename = f"{dataset}_{eval_id}.wav"
            output_path = OUTPUT_DIR / filename

            sf.write(output_path, waveform, sample_rate)

            manifest.append({
                "eval_id": eval_id,
                "dataset": dataset,
                "record_id": row["record_id"],
                "source_shard": shard_name,
                "source_row": source_row,
                "audio_file": filename,
                "sample_rate": sample_rate,
                "audio_duration_seconds": len(waveform) / sample_rate,
                "reference": row["reference"],
                "prediction": row["prediction"],
                "human_label": "",
                "reviewer_notes": "",
            })

            print(f"Extracted {len(manifest)}/{len(merged)}: {filename}")

        except Exception as exc:
            failed.append({
                "eval_id": eval_id,
                "dataset": dataset,
                "error": str(exc),
            })
            print(f"FAILED {dataset} {eval_id}: {exc}")

    manifest_path = OUTPUT_DIR / "review_manifest.csv"
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)

    if failed:
        failed_path = OUTPUT_DIR / "extraction_failures.csv"
        pd.DataFrame(failed).to_csv(failed_path, index=False)
        print(f"\nFailures saved to: {failed_path}")

    print("\nExtraction complete")
    print(f"Sampled: {len(merged)}")
    print(f"Audio created: {len(manifest)}")
    print(f"Failed: {len(failed)}")
    print(f"Audio folder: {OUTPUT_DIR}")
    print(f"Review manifest: {manifest_path}")


if __name__ == "__main__":
    main()