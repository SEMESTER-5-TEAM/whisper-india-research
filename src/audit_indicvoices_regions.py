import pandas as pd
from datasets import load_dataset, Audio
from pathlib import Path

TARGET_STATES = [
    "West Bengal",
    "Jammu & Kashmir",
    "Tamil Nadu",
    "Assam",
]

OUTPUT_FILE = Path(
    "results/indicvoices_hindi_region_audit.csv"
)
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

print("Loading IndicVoices Hindi TRAIN split in streaming mode...")

ds = load_dataset(
    "ai4bharat/IndicVoices",
    "hindi",
    split="train",
    streaming=True,
    token=True,
)

# Do not decode audio; we only need metadata.
ds = ds.cast_column("audio_filepath", Audio(decode=False))

# Normalize state names for matching.
def normalize_state(value):
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())

target_lookup = {
    normalize_state(state): state
    for state in TARGET_STATES
}

stats = {
    state: {
        "recordings": 0,
        "speakers": set(),
        "duration_seconds": 0.0,
        "missing_duration": 0,
        "missing_speaker_id": 0,
    }
    for state in TARGET_STATES
}

rows_seen = 0

for row in ds:
    rows_seen += 1

    state_key = normalize_state(row.get("state"))

    if state_key not in target_lookup:
        continue

    state = target_lookup[state_key]
    item = stats[state]

    item["recordings"] += 1

    speaker = row.get("speaker_id")
    if speaker is not None and str(speaker).strip():
        item["speakers"].add(str(speaker).strip())
    else:
        item["missing_speaker_id"] += 1

    duration = row.get("duration")
    if duration is not None:
        try:
            item["duration_seconds"] += float(duration)
        except (TypeError, ValueError):
            item["missing_duration"] += 1
    else:
        item["missing_duration"] += 1

    # Progress update every 50,000 records.
    if rows_seen % 50000 == 0:
        print(
            f"Scanned {rows_seen:,} rows | "
            f"Matched target-region recordings: "
            f"{sum(x['recordings'] for x in stats.values()):,}"
        )

    # Save interim results so progress isn't lost.
    if rows_seen % 50000 == 0:
        interim = []
        for s, v in stats.items():
            interim.append({
                "state": s,
                "recordings": v["recordings"],
                "unique_speakers_observed": len(v["speakers"]),
                "speech_hours": v["duration_seconds"] / 3600,
                "missing_duration": v["missing_duration"],
                "missing_speaker_id": v["missing_speaker_id"],
                "rows_scanned": rows_seen,
                "status": "INTERIM",
            })
        pd.DataFrame(interim).to_csv(OUTPUT_FILE, index=False)

# Final output
results = []

for state, item in stats.items():
    results.append({
        "state": state,
        "recordings": item["recordings"],
        "unique_speakers_observed": len(item["speakers"]),
        "speech_hours": item["duration_seconds"] / 3600,
        "missing_duration": item["missing_duration"],
        "missing_speaker_id": item["missing_speaker_id"],
        "rows_scanned": rows_seen,
        "status": "COMPLETE",
    })

result_df = pd.DataFrame(results)
result_df.to_csv(OUTPUT_FILE, index=False)

print("\nREGIONAL AUDIT COMPLETE")
print(result_df.to_string(index=False))
print(f"\nSaved to: {OUTPUT_FILE}")