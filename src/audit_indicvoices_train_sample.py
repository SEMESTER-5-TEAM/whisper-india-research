
import duckdb
import pandas as pd
from huggingface_hub import get_token

REPO = "ai4bharat/IndicVoices"
REV = "c96f9088f138cf89d419da7e8e643e1f05c00a87"

SHARDS = {
    "assamese": "assamese/train-00000-of-00112.parquet",
    "bengali": "bengali/train-00000-of-00096.parquet",
    "kashmiri": "kashmiri/train-00000-of-00021.parquet",
    "urdu": "urdu/train-00000-of-00066.parquet",
    "tamil": "tamil/train-00000-of-00118.parquet",
}

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

    for config, path in SHARDS.items():
        url = f"hf://datasets/{REPO}@{REV}/{path}"
        print(f"\nChecking {path}")

        query = """
            SELECT
                state,
                COUNT(*) AS recordings,
                COUNT(DISTINCT speaker_id) AS speakers,
                ROUND(SUM(duration) / 3600.0, 3) AS speech_hours
            FROM read_parquet(?)
            GROUP BY state
            ORDER BY recordings DESC
        """

        rows = con.execute(query, [url]).fetchall()

        for state, recordings, speakers, hours in rows:
            results.append({
                "configuration": config,
                "state": state,
                "recordings_in_shard": recordings,
                "speakers_in_shard": speakers,
                "speech_hours_in_shard": hours,
            })

        print(pd.DataFrame(
            rows,
            columns=[
                "state", "recordings",
                "speakers", "speech_hours"
            ]
        ).to_string(index=False))

    print("\nCOMBINED SAMPLE SUMMARY")
    print(pd.DataFrame(results).to_string(index=False))

finally:
    con.close()
