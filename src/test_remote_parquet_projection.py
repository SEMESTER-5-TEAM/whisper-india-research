
import duckdb
from huggingface_hub import get_token

REVISION = "c96f9088f138cf89d419da7e8e643e1f05c00a87"

# Use DuckDB's Hugging Face protocol, not a regular HTTPS URL.
url = (
    "hf://datasets/ai4bharat/IndicVoices@"
    + REVISION
    + "/hindi/train-00000-of-00082.parquet"
)

token = get_token()
con = duckdb.connect()

try:
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")

    if not token:
        raise RuntimeError(
            "No Hugging Face token found. Check your hf auth login."
        )

    safe_token = token.replace("'", "''")
    con.execute(
        "CREATE SECRET hf_secret "
        f"(TYPE huggingface, TOKEN '{safe_token}')"
    )

    print("Testing one remote training shard...")
    print("Requesting only metadata columns.")

    result = con.execute(
        """
        SELECT state, speaker_id, duration, lang
        FROM read_parquet(?)
        LIMIT 10
        """,
        [url],
    ).fetchdf()

    print("\nSuccess!")
    print("Columns:", list(result.columns))
    print("Rows returned:", len(result))
    print(result.to_string(index=False))

except Exception as exc:
    print("\nTest failed:")
    print(type(exc).__name__ + ":", exc)

finally:
    con.close()
