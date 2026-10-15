
import re
import time
import duckdb
import pandas as pd
from pathlib import Path
from huggingface_hub import get_token

REVISION = "c96f9088f138cf89d419da7e8e643e1f05c00a87"
NUM_SHARDS = 82

TARGETS = {
    "westbengal": "West Bengal",
    "jammuandkashmir": "Jammu & Kashmir",
    "tamilnadu": "Tamil Nadu",
    "assam": "Assam",
}

OUT = Path("results/indicvoices_candidate_region_audit.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

def normalize_state(value):
    return re.sub(r"[^a-z]", "", str(value).lower())

token = get_token()
if not token:
    raise RuntimeError("No Hugging Face token found.")

con = duckdb.connect()

stats = {
    name: {
        "recordings": 0,
        "seconds": 0.0,
        "speakers": set(),
    }
    for name in TARGETS.values()
}

completed_shards = 0

try:
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")

    safe_token = token.replace("'", "''")
    con.execute(
        "CREATE SECRET hf_secret "
        f"(TYPE huggingface, TOKEN '{safe_token}')"
    )

    for i in range(NUM_SHARDS):
        filename = f"hindi/train-{i:05d}-of-{NUM_SHARDS:05d}.parquet"
        url = (
            "hf://datasets/ai4bharat/IndicVoices@"
            + REVISION + "/" + filename
        )

        print(f"\n[{i + 1}/{NUM_SHARDS}] Checking {filename} ...")

        try:
            query = """
                SELECT state, speaker_id, duration
                FROM read_parquet(?)
                WHERE regexp_matches(
                    lower(trim(state)),
                    'west bengal|jammu|tamil nadu|assam'
                )
            """

            rows = con.execute(query, [url]).fetchall()

            for state, speaker_id, duration in rows:
                key = normalize_state(state)

                if key not in TARGETS:
                    continue

                name = TARGETS[key]
                stats[name]["recordings"] += 1

                if speaker_id is not None and str(speaker_id).strip():
                    stats[name]["speakers"].add(str(speaker_id).strip())

                if duration is not None:
                    stats[name]["seconds"] += float(duration)

            completed_shards += 1

            snapshot = []
            for name, item in stats.items():
                snapshot.append({
                    "state": name,
                    "recordings": item["recordings"],
                    "unique_speakers": len(item["speakers"]),
                    "speech_hours": round(item["seconds"] / 3600, 3),
                    "completed_shards": completed_shards,
                    "status": "INTERIM",
                })

            pd.DataFrame(snapshot).to_csv(OUT, index=False)

            print(
                f"Completed {completed_shards}/{NUM_SHARDS} shards. "
                f"Candidate recordings found: "
                f"{sum(x['recordings'] for x in stats.values())}"
            )

        except Exception as exc:
            print(f"Shard failed: {type(exc).__name__}: {exc}")
            print("Progress saved. Stop here and report the error.")
            break

        time.sleep(0.2)

    final_rows = []
    for name, item in stats.items():
        final_rows.append({
            "state": name,
            "recordings": item["recordings"],
            "unique_speakers": len(item["speakers"]),
            "speech_hours": round(item["seconds"] / 3600, 3),
            "completed_shards": completed_shards,
            "status": (
                "COMPLETE"
                if completed_shards == NUM_SHARDS
                else "PARTIAL"
            ),
        })

    pd.DataFrame(final_rows).to_csv(OUT, index=False)

    print("\nREGIONAL AUDIT SUMMARY")
    print(pd.DataFrame(final_rows).to_string(index=False))
    print(f"\nSaved to: {OUT}")

finally:
    con.close()
