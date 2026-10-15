
import duckdb
import pandas as pd
from pathlib import Path
from huggingface_hub import get_token

REPO_ID = "ai4bharat/IndicVoices"
REVISION = "c96f9088f138cf89d419da7e8e643e1f05c00a87"

CONFIGS = {
    "assamese": 2,
    "bengali": 1,
    "kashmiri": 1,
    "urdu": 1,
    "tamil": 2,
}

OUT = Path("results/indicvoices_validation_state_audit.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

token = get_token()
if not token:
    raise RuntimeError("Hugging Face token not found.")

con = duckdb.connect()
results = []

try:
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")

    safe_token = token.replace("'", "''")
    con.execute(
        "CREATE SECRET hf_secret "
        f"(TYPE huggingface, TOKEN '{safe_token}')"
    )

    for config, num_shards in CONFIGS.items():
        for i in range(num_shards):
            filename = (
                f"{config}/valid-{i:05d}-of-{num_shards:05d}.parquet"
            )
            url = f"hf://datasets/{REPO_ID}@{REVISION}/{filename}"

            print(f"\nInspecting metadata: {filename}")

            query = """
                SELECT
                    state,
                    COUNT(*) AS recordings,
                    COUNT(DISTINCT speaker_id) AS unique_speakers,
                    SUM(duration) AS total_seconds
                FROM read_parquet(?)
                GROUP BY state
                ORDER BY recordings DESC
            """

            rows = con.execute(query, [url]).fetchall()

            for state, recordings, speakers, seconds in rows:
                results.append({
                    "configuration": config,
                    "state": state,
                    "recordings": recordings,
                    "unique_speakers": speakers,
                    "speech_hours": round((seconds or 0) / 3600, 3),
                })

            print(f"States found in this shard: {len(rows)}")

    df = pd.DataFrame(results)
    df.to_csv(OUT, index=False)

    print("\nVALIDATION METADATA SUMMARY")
    if not df.empty:
        print(
            df.groupby("configuration")
            .agg(
                states=("state", "nunique"),
                recordings=("recordings", "sum"),
                speakers_across_state_counts=("unique_speakers", "sum"),
            )
            .to_string()
        )

        print("\nState-level results:")
        print(df.to_string(index=False))

    print(f"\nSaved to: {OUT}")

finally:
    con.close()
