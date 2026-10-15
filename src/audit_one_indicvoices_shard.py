
import duckdb
from huggingface_hub import get_token

REVISION = "c96f9088f138cf89d419da7e8e643e1f05c00a87"

url = (
    "hf://datasets/ai4bharat/IndicVoices@"
    + REVISION
    + "/hindi/train-00000-of-00082.parquet"
)

token = get_token()
if not token:
    raise RuntimeError("No Hugging Face token found.")

con = duckdb.connect()

try:
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")

    safe_token = token.replace("'", "''")
    con.execute(
        "CREATE SECRET hf_secret "
        f"(TYPE huggingface, TOKEN '{safe_token}')"
    )

    print("Aggregating metadata from ONE training shard...")
    print("Audio is not selected by the query.")

    query = """
        SELECT
            state,
            COUNT(*) AS recordings,
            COUNT(DISTINCT speaker_id) AS unique_speakers,
            ROUND(SUM(duration) / 3600.0, 2) AS speech_hours
        FROM read_parquet(?)
        GROUP BY state
        ORDER BY recordings DESC
    """

    result = con.execute(query, [url]).fetchdf()

    print("\nRegional metadata for this shard:")
    print(result.to_string(index=False))

except Exception as exc:
    print("\nAudit failed:")
    print(type(exc).__name__ + ":", exc)

finally:
    con.close()
