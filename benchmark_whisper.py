import io
import time
from pathlib import Path

import pandas as pd
import torch
import whisper
import soundfile as sf


# ============================================================
# SETTINGS
# ============================================================

PARQUET_FILE = Path(
    "data/raw/svarah/data/test-00000-of-00003.parquet"
)

NUM_SAMPLES = 10
MODEL_NAME = "small"


# ============================================================
# LANGUAGE MAPPING
# ============================================================

LANGUAGE_MAP = {
    "Hindi": "hi",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Odia": "or",
    "Telugu": "te",
    "Tamil": "ta",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Marathi": "mr",
    "Assamese": "as",
    "Punjabi": "pa",
    "Urdu": "ur",
    "Nepali": "ne",
    "Sindhi": "sd",
}


# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("====================================")
print("LANGUAGE-AWARE WHISPER BENCHMARK")
print("====================================")
print(f"Device: {device}")
print(f"Model: {MODEL_NAME}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# LOAD PARQUET
# ============================================================

print("\nLoading Parquet shard...")

df = pd.read_parquet(PARQUET_FILE)

print(f"Total records in shard: {len(df)}")


# ============================================================
# SELECT SUPPORTED LANGUAGES
# ============================================================

df = df[
    df["primary_language"].isin(LANGUAGE_MAP.keys())
].reset_index(drop=True)

print(
    f"Records with explicitly supported Whisper language: "
    f"{len(df)}"
)


# ============================================================
# LOAD WHISPER
# ============================================================

print("\nLoading Whisper model...")

model = whisper.load_model(
    MODEL_NAME,
    device=device
)

print("Model loaded successfully.")


# ============================================================
# RUN BENCHMARK
# ============================================================

times = []
successful = 0

print("\nRunning inference...\n")

for i in range(min(NUM_SAMPLES, len(df))):

    row = df.iloc[i]

    language_name = row["primary_language"]
    language_code = LANGUAGE_MAP[language_name]

    audio_obj = row["audio_filepath"]

    if isinstance(audio_obj, dict):
        audio_bytes = audio_obj["bytes"]
    else:
        raise TypeError(
            "Unexpected audio_filepath format."
        )

    audio_array, sample_rate = sf.read(
        io.BytesIO(audio_bytes),
        dtype="float32"
    )

    start = time.perf_counter()

    result = model.transcribe(
        audio_array,
        language=language_code,
        fp16=(device == "cuda"),
        temperature=0
    )

    elapsed = time.perf_counter() - start

    times.append(elapsed)
    successful += 1

    print(
        f"{i + 1:02d}/{NUM_SAMPLES} | "
        f"language={language_name} ({language_code}) | "
        f"duration={row['duration']:.2f}s | "
        f"inference={elapsed:.2f}s"
    )

    print(
        f"    Reference: {str(row['text'])[:120]}"
    )

    print(
        f"    Whisper:   {result['text'][:120]}"
    )


# ============================================================
# RESULTS
# ============================================================

print("\n====================================")
print("BENCHMARK COMPLETE")
print("====================================")

print(f"Successful: {successful}/{NUM_SAMPLES}")

if times:

    avg_time = sum(times) / len(times)

    avg_duration = (
        df.iloc[:successful]["duration"].mean()
    )

    print(
        f"Average inference time: "
        f"{avg_time:.2f} sec/sample"
    )

    print(
        f"Average audio duration: "
        f"{avg_duration:.2f} sec"
    )

    print(
        f"Estimated time for 1,000 recordings: "
        f"{avg_time * 1000 / 3600:.2f} hours"
    )

if device == "cuda":

    print(
        f"Peak GPU memory: "
        f"{torch.cuda.max_memory_allocated() / (1024**3):.2f} GB"
    )