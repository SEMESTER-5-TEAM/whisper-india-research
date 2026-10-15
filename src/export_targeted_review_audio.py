#!/usr/bin/env python
"""Export audio clips for the targeted human-validation sheet.

Run from project root:
    python src\export_targeted_review_audio.py

Reads only the 30 recording keys in:
    results/lora_generic_hindi/targeted_human_validation_sheet.csv

Looks up each recording by source_shard and source_row in data/raw/lahaja/data,
then writes WAV files named by review_id to:
    results/lora_generic_hindi/targeted_review_audio/

Requires pandas, pyarrow (via pandas parquet support), and soundfile.
"""
from pathlib import Path
import io
import re
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "lahaja" / "data"
RESULTS = ROOT / "results" / "lora_generic_hindi"
SHEET = RESULTS / "targeted_human_validation_sheet.csv"
OUT_DIR = RESULTS / "targeted_review_audio"


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def extract_audio_payload(audio_value):
    if isinstance(audio_value, dict):
        payload = audio_value.get("bytes")
        if payload is not None:
            return payload
    # pandas may return a pyarrow struct-like object
    try:
        payload = audio_value["bytes"]
        if payload is not None:
            return payload
    except Exception:
        pass
    raise ValueError("Could not find embedded audio bytes in audio_filepath struct")


def main():
    if not SHEET.exists():
        raise FileNotFoundError(f"Missing review sheet: {SHEET}")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Missing Parquet directory: {DATA_DIR}")

    sheet = pd.read_csv(SHEET, keep_default_na=False)
    required = {"review_id", "source_shard", "source_row"}
    missing = required - set(sheet.columns)
    if missing:
        raise ValueError(f"Review sheet missing columns: {sorted(missing)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_by_name = {p.name: p for p in DATA_DIR.glob("*.parquet")}
    missing_shards = sorted(set(sheet["source_shard"].astype(str)) - set(parquet_by_name))
    if missing_shards:
        raise FileNotFoundError(f"Could not locate these source shards: {missing_shards}")

    # Read each needed shard once, then export its selected rows.
    exported = 0
    for shard_name, group in sheet.groupby("source_shard", sort=False):
        parquet_path = parquet_by_name[str(shard_name)]
        data = pd.read_parquet(parquet_path, columns=["audio_filepath"])
        for row in group.itertuples(index=False):
            try:
                row_index = int(row.source_row)
            except Exception as exc:
                raise ValueError(f"Invalid source_row for {row.review_id}: {row.source_row}") from exc
            if row_index < 0 or row_index >= len(data):
                raise IndexError(
                    f"{row.review_id}: source_row {row_index} outside shard with {len(data)} rows"
                )
            audio_value = data.iloc[row_index]["audio_filepath"]
            payload = extract_audio_payload(audio_value)
            if not isinstance(payload, (bytes, bytearray, memoryview)):
                raise TypeError(f"{row.review_id}: embedded audio payload is not bytes")
            # Decode the original audio container, then save a review-friendly WAV.
            audio, sample_rate = sf.read(io.BytesIO(bytes(payload)), dtype="float32", always_2d=False)
            if getattr(audio, "ndim", 1) > 1:
                audio = audio.mean(axis=1)
            out_path = OUT_DIR / f"{safe_name(row.review_id)}.wav"
            sf.write(out_path, audio, sample_rate, subtype="PCM_16")
            exported += 1

    print(f"Exported {exported} audio clips to:")
    print(OUT_DIR.relative_to(ROOT))
    print("Match each WAV filename (for example, TV001.wav) to review_id in the CSV.")
    if exported != len(sheet):
        print(f"WARNING: exported {exported} of {len(sheet)} requested clips.")


if __name__ == "__main__":
    main()
