
import time
import duckdb
import pandas as pd
from pathlib import Path
from huggingface_hub import get_token

REPO = "ai4bharat/IndicVoices"
REVISION = "c96f9088f138cf89d419da7e8e643e1f05c00a87"

# Configuration, number of training shards, target state
TARGETS = {
    "assamese": (112, "Assam"),
    "bengali": (96, "West Bengal"),
    "kashmiri": (21, "Jammu Kashmir"),
    "tamil": (118, "Tamil Nadu"),
}

OUT_DIR = Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_FILE = OUT_DIR / "indicvoices_training_audit_by_shard.csv"
SUMMARY_FILE = OUT_DIR / "indicvoices_training_region_totals.csv"

token = get_token()
if not token:
    raise RuntimeError("Hugging Face token not found.")

con = duckdb.connect()

# Keep speaker IDs in sets so speakers appearing in multiple shards
# are counted only once in the final total.
totals = {
    config: {
        "recordings": 0,
        "seconds": 0.0,
        "speakers": set(),
    }
    for config in TARGETS
}

detail_rows = []
completed = 0
total_shards = sum(n for n, _ in TARGETS.values())

def save_progress():
    pd.DataFrame(detail_rows).to_csv(DETAIL_FILE, index=False)

    summary_rows = []
    for config, item in totals.items():
        summary_rows.append({
            "configuration": config,
            "target_state": TARGETS[config][1],
            "recordings": item["recordings"],
            "unique_speakers": len(item["speakers"]),
            "speech_hours": round(item["seconds"] / 3600, 3),
            "completed_shards": completed,
            "total_shards": total_shards,
            "status": (
                "COMPLETE" if completed == total_shards else "PARTIAL"
            ),
        })

    pd.DataFrame(summary_rows).to_csv(SUMMARY_FILE, index=False)

try:
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")

    safe_token = token.replace("'", "''")
    con.execute(
        "CREATE SECRET hf_secret "
        f"(TYPE huggingface, TOKEN '{safe_token}')"
    )

    for config, (num_shards, target_state) in TARGETS.items():
        for i in range(num_shards):
            filename = (
                f"{config}/train-{i:05d}-of-{num_shards:05d}.parquet"
            )
            url = (
                f"hf://datasets/{REPO}@{REVISION}/{filename}"
            )

            print(
                f"\n[{completed + 1}/{total_shards}] "
                f"{filename} | target: {target_state}",
                flush=True,
            )

            query = """
                SELECT
                    speaker_id,
                    COUNT(*) AS recordings,
                    SUM(duration) AS seconds
                FROM read_parquet(?)
                WHERE lower(trim(state)) = lower(?)
                GROUP BY speaker_id
            """

            rows = con.execute(
                query, [url, target_state]
            ).fetchall()

            shard_recordings = 0
            shard_seconds = 0.0
            shard_speakers = set()

            for speaker_id, recordings, seconds in rows:
                shard_recordings += int(recordings)
                shard_seconds += float(seconds or 0)

                if speaker_id is not None and str(speaker_id).strip():
                    shard_speakers.add(str(speaker_id).strip())
                    totals[config]["speakers"].add(
                        str(speaker_id).strip()
                    )

            totals[config]["recordings"] += shard_recordings
            totals[config]["seconds"] += shard_seconds

            detail_rows.append({
                "configuration": config,
                "target_state": target_state,
                "shard": filename,
                "recordings": shard_recordings,
                "speakers_in_shard": len(shard_speakers),
                "speech_hours": round(shard_seconds / 3600, 3),
            })

            completed += 1
            save_progress()

            print(
                f"Target recordings in shard: {shard_recordings}; "
                f"progress: {completed}/{total_shards}",
                flush=True,
            )

            time.sleep(0.1)

except Exception as exc:
    print(f"\nAUDIT STOPPED: {type(exc).__name__}: {exc}")
    print("Progress has been saved for completed shards.")
    raise

finally:
    save_progress()
    con.close()

print("\nFINAL TRAINING-DATA AUDIT")
print(pd.read_csv(SUMMARY_FILE).to_string(index=False))
print(f"\nShard details: {DETAIL_FILE}")
print(f"Regional totals: {SUMMARY_FILE}")
