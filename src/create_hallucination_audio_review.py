import os
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
import numpy as np

# ============================================================
# SETTINGS
# ============================================================

PROJECT_DIR = r"C:\Users\divya\whisper-india-project"

REVIEW_CSV = os.path.join(
    PROJECT_DIR,
    "results",
    "baseline",
    "hallucination_review.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "baseline",
    "review_audio"
)

SVARAH_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "raw",
    "svarah",
    "data"
)

LAHAJA_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "raw",
    "lahaja",
    "data"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD REVIEW FILE
# ============================================================

df = pd.read_csv(REVIEW_CSV)

print("=" * 60)
print("CREATING AUDIO FILES FOR HUMAN HALLUCINATION REVIEW")
print("=" * 60)
print(f"Candidates: {len(df)}")

# ============================================================
# CACHE PARQUET DATA
# ============================================================

parquet_cache = {}

def load_shard(dataset, shard_name):
    key = (dataset, shard_name)

    if key not in parquet_cache:

        if dataset.lower() == "svarah":
            path = os.path.join(SVARAH_DIR, shard_name)

        elif dataset.lower() == "lahaja":
            path = os.path.join(LAHAJA_DIR, shard_name)

        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        print(f"Loading: {path}")

        parquet_cache[key] = pq.read_table(path).to_pandas()

    return parquet_cache[key]


# ============================================================
# EXTRACT AUDIO
# ============================================================

created = 0
failed = 0

for i, row in df.iterrows():

    try:
        dataset = row["dataset"]
        eval_id = int(row["eval_id"])

        # ----------------------------------------------------
        # The baseline eval_id is based on the combined
        # evaluation CSV. We need the original source row.
        # The candidate CSV contains source information only
        # indirectly, so locate the matching record by
        # reference + prediction + dataset + state.
        # ----------------------------------------------------

        reference = str(row["reference"])
        prediction = str(row["prediction"])

        found = False

        if dataset.lower() == "svarah":

            shards = [
                "test-00000-of-00003.parquet",
                "test-00001-of-00003.parquet",
                "test-00002-of-00003.parquet",
            ]

        elif dataset.lower() == "lahaja":

            shards = [
                "test-00000-of-00004.parquet",
                "test-00001-of-00004.parquet",
                "test-00002-of-00004.parquet",
                "test-00003-of-00004.parquet",
            ]

        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        # ----------------------------------------------------
        # Find matching reference in source data.
        # ----------------------------------------------------

        for shard_name in shards:

            data = load_shard(dataset, shard_name)

            for source_row, source in data.iterrows():

                source_text = (
                    str(source["text"])
                    if "text" in source
                    else str(source["normalized"])
                )

                if source_text == reference:

                    audio = source["audio_filepath"]

                    # HuggingFace Audio feature is normally:
                    # {"bytes": ..., "path": ...}

                    if isinstance(audio, dict):
                        audio_bytes = audio.get("bytes")

                    else:
                        audio_bytes = None

                    if audio_bytes is None:
                        continue

                    # Convert raw audio bytes into waveform.
                    import io

                    audio_buffer = io.BytesIO(audio_bytes)

                    waveform, sample_rate = sf.read(
                        audio_buffer,
                        dtype="float32"
                    )

                    # Stereo -> mono
                    if waveform.ndim > 1:
                        waveform = np.mean(
                            waveform,
                            axis=1
                        )

                    filename = (
                        f"{i:03d}_"
                        f"{dataset}_"
                        f"{eval_id}.wav"
                    )

                    output_path = os.path.join(
                        OUTPUT_DIR,
                        filename
                    )

                    sf.write(
                        output_path,
                        waveform,
                        sample_rate
                    )

                    created += 1

                    found = True

                    print(
                        f"[{created:03d}/{len(df)}] "
                        f"{filename}"
                    )

                    break

            if found:
                break

        if not found:
            print(
                f"WARNING: Could not find audio "
                f"for candidate {i}"
            )
            failed += 1

    except Exception as e:

        print(
            f"ERROR candidate {i}: {e}"
        )

        failed += 1


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("AUDIO EXTRACTION COMPLETE")
print("=" * 60)

print(f"Created: {created}")
print(f"Failed : {failed}")

print()
print("Audio directory:")
print(OUTPUT_DIR)