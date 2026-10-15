import pandas as pd
import io
import soundfile as sf
import whisper
import torch
import time
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "small"
DEVICE = "cuda"

INPUT_FILE = Path(
    "data/metadata/evaluation_subset_60_per_state.csv"
)

OUTPUT_DIR = Path("results/baseline")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = OUTPUT_DIR / "whisper_small_baseline_expanded.csv"

CHECKPOINT_EVERY = 10


# ============================================================
# Load evaluation metadata
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("WHISPER BASELINE EVALUATION")
print("=" * 60)

print(f"Total evaluation records: {len(df)}")
print(f"Model: Whisper {MODEL_NAME}")
print(f"Device: {DEVICE}")
print("FP16: False")
print("Language: automatic detection")
print("Temperature: 0")


# ============================================================
# Load previous results if they exist
# ============================================================

# Previous 1,050-record baseline
OLD_OUTPUT_FILE = Path(
    "results/baseline/whisper_small_baseline.csv"
)

if OUTPUT_FILE.exists():

    results = pd.read_csv(
        OUTPUT_FILE
    )

    print(
        f"\nExpanded results found: "
        f"{len(results)}"
    )

elif OLD_OUTPUT_FILE.exists():

    print(
        "\nNo expanded results found."
    )

    print(
        "Loading existing 1,050-record baseline..."
    )

    results = pd.read_csv(
        OLD_OUTPUT_FILE
    )

    # Save a copy as the starting point
    results.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print(
        f"Loaded existing baseline: "
        f"{len(results)} records"
    )

else:

    results = pd.DataFrame()

    print(
        "\nNo previous results found."
    )


completed_ids = set()

if not results.empty:

    completed_ids = set(
        results["eval_id"].astype(int)
    )

print(
    f"Already completed: "
    f"{len(completed_ids)}"
)


# ============================================================
# Load Whisper
# ============================================================

print("\nLoading Whisper model...")

model = whisper.load_model(
    MODEL_NAME,
    device=DEVICE
)

print("Whisper model loaded.")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"GPU memory allocated: "
        f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )


# ============================================================
# Helper function
# ============================================================

def decode_audio(audio_filepath):

    """
    Decode embedded HF audio object.

    Svarah and LAHAJA store audio inside
    the Parquet files rather than as ordinary
    standalone WAV files.
    """

    if not isinstance(audio_filepath, dict):

        raise TypeError(
            f"Expected audio dictionary, got "
            f"{type(audio_filepath)}"
        )

    audio_bytes = audio_filepath["bytes"]

    audio, sample_rate = sf.read(
        io.BytesIO(audio_bytes),
        dtype="float32"
    )

    return audio, sample_rate


# ============================================================
# Process recordings
# ============================================================

start_time = time.time()

new_results = []

total = len(df)

for position, row in df.iterrows():

    eval_id = int(row["eval_id"])

    # --------------------------------------------------------
    # Resume support
    # --------------------------------------------------------

    if eval_id in completed_ids:

        continue

    record_start = time.time()

    try:

        # ----------------------------------------------------
        # Locate original Parquet file
        # ----------------------------------------------------

        dataset = row["dataset"]

        source_shard = row["source_shard"]

        source_row = int(row["source_row"])

        if dataset == "svarah":

            parquet_file = (
                Path("data/raw/svarah/data")
                / source_shard
            )

        elif dataset == "lahaja":

            parquet_file = (
                Path("data/raw/lahaja/data")
                / source_shard
            )

        else:

            raise ValueError(
                f"Unknown dataset: {dataset}"
            )


        # ----------------------------------------------------
        # Read only the required source row
        # ----------------------------------------------------

        source_df = pd.read_parquet(
            parquet_file
        )

        audio_object = source_df.iloc[
            source_row
        ]["audio_filepath"]


        # ----------------------------------------------------
        # Decode audio
        # ----------------------------------------------------

        audio, sample_rate = decode_audio(
            audio_object
        )


        # ----------------------------------------------------
        # Whisper transcription
        # ----------------------------------------------------

        result = model.transcribe(
            audio,
            fp16=False,
            temperature=0
        )


        prediction = result["text"].strip()

        detected_language = result.get(
            "language",
            ""
        )


        # ----------------------------------------------------
        # Basic audio information
        # ----------------------------------------------------

        audio_duration = len(audio) / sample_rate


        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        result_row = {

            "eval_id": eval_id,

            "dataset": dataset,

            "record_id": row["record_id"],

            "source_shard": source_shard,

            "source_row": source_row,

            "state": (
                row["native_place_state"]
                if dataset == "svarah"
                else row["native_state"]
            ),

            "district": (
                row["native_place_district"]
                if dataset == "svarah"
                else row["native_district"]
            ),

            "primary_language": (
                row["primary_language"]
                if dataset == "svarah"
                else ""
            ),

            "native_language": (
                row["native_language"]
                if dataset == "lahaja"
                else ""
            ),

            "gender": row["gender"],

            "age_group": (
                row["age-group"]
                if dataset == "svarah"
                else row["age_group"]
            ),

            "reference": row["text"],

            "prediction": prediction,

            "detected_language": detected_language,

            "audio_duration": audio_duration,

            "sample_rate": sample_rate,

            "processing_seconds": (
                time.time() - record_start
            ),

            "status": "success"
        }


    except Exception as e:

        # ----------------------------------------------------
        # Do not lose the record if one sample fails
        # ----------------------------------------------------

        result_row = {

            "eval_id": eval_id,

            "dataset": row["dataset"],

            "record_id": row["record_id"],

            "source_shard": row["source_shard"],

            "source_row": row["source_row"],

            "state": (
                row["native_place_state"]
                if row["dataset"] == "svarah"
                else row["native_state"]
            ),

            "district": (
                row["native_place_district"]
                if row["dataset"] == "svarah"
                else row["native_district"]
            ),

            "primary_language": (
                row["primary_language"]
                if row["dataset"] == "svarah"
                else ""
            ),

            "native_language": (
                row["native_language"]
                if row["dataset"] == "lahaja"
                else ""
            ),

            "gender": row["gender"],

            "age_group": (
                row["age-group"]
                if row["dataset"] == "svarah"
                else row["age_group"]
            ),

            "reference": row["text"],

            "prediction": "",

            "detected_language": "",

            "audio_duration": row["duration"],

            "sample_rate": "",

            "processing_seconds": (
                time.time() - record_start
            ),

            "status": f"ERROR: {repr(e)}"
        }


        print(
            f"\nERROR | eval_id={eval_id} | {repr(e)}"
        )


    new_results.append(
        result_row
    )


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    completed_now = (
        len(completed_ids)
        + len(new_results)
    )

    elapsed = time.time() - start_time

    avg_time = (
        elapsed / len(new_results)
        if new_results
        else 0
    )

    remaining = total - completed_now

    estimated_remaining = (
        remaining * avg_time
    )


    print(
        f"[{completed_now}/{total}] "
        f"{dataset} | "
        f"state={result_row['state']} | "
        f"lang={result_row['detected_language']} | "
        f"time={result_row['processing_seconds']:.2f}s"
    )


    # --------------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------------

    if (
        len(new_results) % CHECKPOINT_EVERY == 0
    ):

        checkpoint = pd.DataFrame(
            new_results
        )

        combined = pd.concat(
            [results, checkpoint],
            ignore_index=True
        )

        combined = (
            combined
            .drop_duplicates(
                subset=["eval_id"],
                keep="last"
            )
            .sort_values("eval_id")
        )

        combined.to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8"
        )

        print(
            f"\nCheckpoint saved: "
            f"{len(combined)} records"
        )

        print(
            f"Estimated remaining time: "
            f"{estimated_remaining / 60:.1f} minutes\n"
        )

        results = combined
        new_results = []


# ============================================================
# Final save
# ============================================================

if new_results:

    checkpoint = pd.DataFrame(
        new_results
    )

    combined = pd.concat(
        [results, checkpoint],
        ignore_index=True
    )

else:

    combined = results


combined = (
    combined
    .drop_duplicates(
        subset=["eval_id"],
        keep="last"
    )
    .sort_values("eval_id")
)

combined.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# ============================================================
# Final report
# ============================================================

elapsed_total = time.time() - start_time

print("\n")
print("=" * 60)
print("BASELINE COMPLETE")
print("=" * 60)

print(
    f"Results saved: {OUTPUT_FILE}"
)

print(
    f"Total results: {len(combined)}"
)

print(
    f"Successful: "
    f"{(combined['status'] == 'success').sum()}"
)

print(
    f"Errors: "
    f"{(combined['status'] != 'success').sum()}"
)

print(
    f"Total processing time: "
    f"{elapsed_total / 60:.2f} minutes"
)

print("\nDataset results:")

print(
    combined["dataset"].value_counts()
)

print("\nDetected languages:")

print(
    combined[
        combined["status"] == "success"
    ]["detected_language"]
    .value_counts()
)

print("\nOutput file:")

print(
    OUTPUT_FILE.resolve()
)