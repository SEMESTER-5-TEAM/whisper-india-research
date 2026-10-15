"""
Automatic-language-detection inference for baseline Whisper-small and the existing LoRA model.
Run from the project root:
    python src\\run_default_decoding.py
Smoke test:
    python src\\run_default_decoding.py --max_samples 5

Saves separate files and checkpoints; does not overwrite forced-Hindi predictions.
"""
import argparse
import gc
import io
import time
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL_NAME = "openai/whisper-small"
META = Path("data/metadata/lahaja_speaker_disjoint_split.csv")
ADAPTER_DIR = Path("models/lora_generic_hindi/adapter_epoch_1")
OUT_DIR = Path("results/lora_generic_hindi")
CHECKPOINT_EVERY = 10
MAX_AUDIO_SECONDS = 30


def resolve_shard(value):
    value = str(value).replace("\\", "/")
    return Path(value) if value.startswith("data/") else Path("data/raw/lahaja/data") / value


class AudioReader:
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)
        self.current_path = None
        self.shard = None

    def __len__(self):
        return len(self.frame)

    def get(self, index):
        row = self.frame.iloc[index]
        path = resolve_shard(row["source_shard"])
        if path != self.current_path:
            self.shard = pd.read_parquet(path, columns=["audio_filepath"])
            self.current_path = path
        audio_object = self.shard.iloc[int(row["source_row"])]["audio_filepath"]
        if not isinstance(audio_object, dict) or not audio_object.get("bytes"):
            raise ValueError(f"Missing embedded audio: {path}, row {row['source_row']}")
        audio, sample_rate = sf.read(io.BytesIO(audio_object["bytes"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != 16000:
            raise ValueError(f"Expected 16 kHz audio, got {sample_rate}: {path}")
        return audio[: MAX_AUDIO_SECONDS * sample_rate], sample_rate, row


def output_record(row, prediction="", seconds=None, status="success", error=""):
    return {
        "record_id": row.get("record_id", ""),
        "source_shard": row["source_shard"],
        "source_row": int(row["source_row"]),
        "sp_id": row["sp_id"],
        "state": row.get("native_state", ""),
        "district": row.get("native_district", ""),
        "reference": str(row["verbatim"]),
        "prediction": prediction,
        "processing_seconds": seconds,
        "status": status,
        "error": error,
        "decoding": "automatic_language_detection",
    }


def run_condition(test, processor, model, device, output_path, label, max_samples=0):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        previous = pd.read_csv(output_path)
        required = {"source_shard", "source_row", "status"}
        if not required.issubset(previous.columns):
            raise ValueError(f"Existing output has unexpected schema: {output_path}")
        completed = set(zip(previous["source_shard"].astype(str),
                            previous["source_row"].astype(int)))
        records = previous.to_dict("records")
        print(f"{label}: resuming from {len(completed)} existing rows.")
    else:
        completed, records = set(), []

    pending = test[~test.apply(
        lambda r: (str(r["source_shard"]), int(r["source_row"])) in completed, axis=1
    )].copy()
    if max_samples:
        pending = pending.head(max_samples).copy()
    print(f"{label}: pending rows this run = {len(pending)}")
    if pending.empty:
        return

    model.eval()
    reader = AudioReader(pending)
    new_records = []
    started = time.time()

    for i in range(len(reader)):
        row = pending.iloc[i]
        t0 = time.time()
        try:
            audio, sample_rate, row = reader.get(i)
            features = processor.feature_extractor(
                audio, sampling_rate=sample_rate, return_tensors="pt"
            ).input_features.to(device=device, dtype=torch.float32)
            with torch.inference_mode():
                token_ids = model.generate(
                    input_features=features,
                    task="transcribe",
                    language=None,
                    do_sample=False,
                    max_new_tokens=225,
                )
            prediction = processor.tokenizer.batch_decode(
                token_ids, skip_special_tokens=True
            )[0].strip()
            rec = output_record(row, prediction, round(time.time() - t0, 4))
            del features, token_ids
        except Exception as exc:
            rec = output_record(
                row, seconds=round(time.time() - t0, 4),
                status="error", error=repr(exc)
            )
            print(f"{label}: ERROR key=({row['source_shard']},{row['source_row']}): {exc}",
                  flush=True)

        new_records.append(rec)
        if (i + 1) % CHECKPOINT_EVERY == 0 or i + 1 == len(reader):
            records.extend(new_records)
            new_records = []
            pd.DataFrame(records).drop_duplicates(
                subset=["source_shard", "source_row"], keep="last"
            ).to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"{label}: {i+1}/{len(reader)} processed this run; "
                  f"saved rows={len(records)}; elapsed={(time.time()-started)/60:.1f} min",
                  flush=True)
    print(f"{label}: saved {output_path}")


def load_model(use_lora, device):
    base = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    if use_lora:
        if not ADAPTER_DIR.exists():
            raise FileNotFoundError(f"LoRA adapter not found: {ADAPTER_DIR}")
        model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    else:
        model = base
    model.to(device)
    model.generation_config.language = None
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Maximum NEW recordings per model this run; 0 means all remaining.")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; activate the whisper-india environment.")
    if not META.exists():
        raise FileNotFoundError(f"Split metadata not found: {META}")

    metadata = pd.read_csv(META)
    required = {"split", "source_shard", "source_row", "verbatim", "sp_id"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Split metadata missing columns: {sorted(missing)}")
    test = metadata[metadata["split"].eq("test")].copy().reset_index(drop=True)
    if test.empty:
        raise ValueError("No test rows found in split metadata.")
    if test.duplicated(["source_shard", "source_row"]).any():
        raise ValueError("Duplicate source keys in test split.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Test recordings:", len(test), "| speakers:", test["sp_id"].nunique())
    print("Decoding: automatic language detection, transcription task, greedy decoding.")
    print("Existing forced-Hindi prediction files will not be modified.")
    device = torch.device("cuda")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)

    conditions = [
        ("baseline_default", False, OUT_DIR / "whisper_small_default_test_predictions.csv"),
        ("lora_default", True, OUT_DIR / "whisper_small_lora_default_test_predictions.csv"),
    ]
    for label, use_lora, output_path in conditions:
        print(f"\\nLoading {'LoRA-adapted' if use_lora else 'baseline'} Whisper-small...")
        model = load_model(use_lora, device)
        run_condition(test, processor, model, device, output_path, label, args.max_samples)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("\\nDone. Default-decoding predictions are saved separately.")
    print("Baseline:", OUT_DIR / "whisper_small_default_test_predictions.csv")
    print("LoRA:", OUT_DIR / "whisper_small_lora_default_test_predictions.csv")


if __name__ == "__main__":
    main()
