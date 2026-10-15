
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "metadata" / "lahaja_metadata.csv"

df = pd.read_csv(INPUT)

# Clean fields used in the audit.
df["native_state"] = df["native_state"].fillna("UNKNOWN").astype(str).str.strip()
df["native_district"] = df["native_district"].fillna("UNKNOWN").astype(str).str.strip()
df["sp_id"] = df["sp_id"].fillna("UNKNOWN").astype(str).str.strip()
df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

print("=" * 75)
print("LAHAJA SPEAKER AND REGION AUDIT")
print("=" * 75)

print(f"Total recordings: {len(df)}")
print(f"Distinct speaker IDs: {df['sp_id'].nunique()}")
print(f"States: {df['native_state'].nunique()}")
print(f"Districts: {df['native_district'].nunique()}")
print(f"Total audio hours: {df['duration'].sum() / 3600:.2f}")
print(f"Missing durations: {df['duration'].isna().sum()}")

print("\n1. RECORDINGS, SPEAKERS, AND HOURS BY STATE")
state_summary = (
    df.groupby("native_state", dropna=False)
    .agg(
        recordings=("record_id", "count"),
        speakers=("sp_id", "nunique"),
        districts=("native_district", "nunique"),
        total_seconds=("duration", "sum"),
        median_duration=("duration", "median"),
    )
    .reset_index()
)

state_summary["audio_hours"] = state_summary["total_seconds"] / 3600
state_summary = state_summary.sort_values(
    ["speakers", "audio_hours"], ascending=False
)

print(
    state_summary[
        ["native_state", "recordings", "speakers", "districts",
         "audio_hours", "median_duration"]
    ].to_string(
        index=False,
        formatters={
            "audio_hours": "{:.2f}".format,
            "median_duration": "{:.2f}".format,
        }
    )
)

print("\n2. RECORDING SCENARIOS")
if "scenario" in df.columns:
    print(df["scenario"].fillna("UNKNOWN").value_counts().to_string())

print("\n3. SPEAKER IDS ASSOCIATED WITH MULTIPLE STATES")
speaker_states = df.groupby("sp_id")["native_state"].nunique()
multi_state_ids = speaker_states[speaker_states > 1]

print(f"Speakers associated with multiple states: {len(multi_state_ids)}")
if len(multi_state_ids):
    print("Examples:")
    print(multi_state_ids.head(10).to_string())

print("\n4. SPEAKER-LEVEL RECORDING DISTRIBUTION")
recordings_per_speaker = df.groupby("sp_id").size()
print(recordings_per_speaker.describe().round(2).to_string())

print("\n5. TRANSCRIPT FIELD AVAILABILITY")
for column in ["verbatim", "normalized", "text"]:
    if column in df.columns:
        text = df[column].fillna("").astype(str).str.strip()
        print(
            f"{column}: non-empty={int(text.ne('').sum())}, "
            f"empty={int(text.eq('').sum())}"
        )

print("\n6. POSSIBLE TARGET REGIONS")
print(
    "Use the table above to identify states with enough distinct speakers "
    "and audio hours. Do not select a target based on test performance."
)

print("\nAudit complete.")
