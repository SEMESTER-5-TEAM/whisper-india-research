import io
import time
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import whisper

MODEL_NAME = "small"
DEVICE = "cuda"

INPUT_FILE = Path(
    "data/metadata/evaluation_subset_60_per_state.csv"
)

BASELINE_FILE = Path(
    "results/baseline/whisper_small_baseline_expanded.csv"
)

OUTPUT_FILE = Path(
    "results/intervention/whisper_small_no_previous_context.csv"
)

CHECKPOINT_EVERY = 10

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)
baseline = pd.read_csv(BASELINE_FILE)

# Only process recordings that have successful baseline predictions.
baseline_success = baseline[
    baseline["status"].eq("success")
].copy()

valid_ids = set(baseline_success["eval_id"].astype(int))
df = df[df["eval_id"].astype(int).isin(valid_ids)].copy()
# Resume safely from this experiment's own output file.
if OUTPUT_FILE.exists():
    results = pd.read_csv(OUTPUT_FILE)
    completed_ids = set(results["eval_id"].astype(int))
else:
    results = pd.DataFrame()
    completed_ids = set()

print("=" * 60)
print("DECODING INTERVENTION")
print("=" * 60)
print("Model:", MODEL_NAME)
print("Device:", DEVICE)
print("FP16: False")
print("Temperature: 0")
print("Language: automatic detection")
print("Condition on previous text: False")
print("Eligible recordings:", len(df))
print("Already completed:", len(completed_ids))
print("Output:", OUTPUT_FILE)

model = whisper.load_model(MODEL_NAME, device=DEVICE)

def decode_audio(audio_filepath):
    if not isinstance(audio_filepath, dict):
        raise TypeError(
            f"Expected embedded audio dictionary, got "
            f"{type(audio_filepath)}"
        )

    audio_bytes = audio_filepath["bytes"]
    audio, sample_rate = sf.read(
        io.BytesIO(audio_bytes),
        dtype="float32",
    )
    return audio, sample_rate


new_results = []
start_time = time.time()

for _, row in df.iterrows():
    eval_id = int(row["eval_id"])

    if eval_id in completed_ids:
        continue

    record_start = time.time()

    try:
        dataset = row["dataset"]
        source_shard = row["source_shard"]
        source_row = int(row["source_row"])

        if dataset == "svarah":
            parquet_file = (
                Path("data/raw/svarah/data") / source_shard
            )
        elif dataset == "lahaja":
            parquet_file = (
                Path("data/raw/lahaja/data") / source_shard
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        source_df = pd.read_parquet(parquet_file)
        audio_object = source_df.iloc[source_row]["audio_filepath"]
        audio, sample_rate = decode_audio(audio_object)

        result = model.transcribe(
            audio,
            fp16=False,
            temperature=0,
            condition_on_previous_text=False,
        )

        prediction = result["text"].strip()

        result_row = {
            "eval_id": eval_id,
            "dataset": dataset,
            "record_id": row["record_id"],
            "state": (
                row["native_place_state"]
                if dataset == "svarah"
                else row["native_state"]
            ),
            "reference": row["text"],
            "prediction": prediction,
            "detected_language": result.get("language", ""),
            "audio_duration": len(audio) / sample_rate,
            "processing_seconds": time.time() - record_start,
            "status": "success",
        }

    except Exception as exc:
        result_row = {
            "eval_id": eval_id,
            "dataset": row["dataset"],
            "record_id": row["record_id"],
            "state": (
                row["native_place_state"]
                if row["dataset"] == "svarah"
                else row["native_state"]
            ),
            "reference": row["text"],
            "prediction": "",
            "detected_language": "",
            "audio_duration": row["duration"],
            "processing_seconds": time.time() - record_start,
            "status": f"ERROR: {repr(exc)}",
        }

        print(f"ERROR | eval_id={eval_id} | {repr(exc)}")

    new_results.append(result_row)

    print(
        f"[{len(completed_ids) + len(new_results)}/"
        f"{len(df)}] {row['dataset']} | "
        f"eval_id={eval_id} | "
        f"{result_row['status']} | "
        f"{result_row['processing_seconds']:.2f}s"
    )

    if len(new_results) % CHECKPOINT_EVERY == 0:
        combined = pd.concat(
            [results, pd.DataFrame(new_results)],
            ignore_index=True,
        )
        combined = (
            combined.drop_duplicates("eval_id", keep="last")
            .sort_values("eval_id")
        )
        combined.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
        results = combined
        new_results = []
        print("Checkpoint saved:", len(results))

if new_results:
    results = pd.concat(
        [results, pd.DataFrame(new_results)],
        ignore_index=True,
    )

results = (
    results.drop_duplicates("eval_id", keep="last")
    .sort_values("eval_id")
)
results.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print("\nINTERVENTION RUN FINISHED")
print("Total output rows:", len(results))
print("Successful:", int(results["status"].eq("success").sum()))
print("Errors:", int((~results["status"].eq("success")).sum()))
print("Saved to:", OUTPUT_FILE.resolve())
print("Elapsed minutes:", round((time.time() - start_time) / 60, 2))