
import pandas as pd
import numpy as np

# Load the complete LAHAJA metadata.
df = pd.read_csv("data/metadata/lahaja_metadata.csv")
df["sp_id"] = pd.to_numeric(df["sp_id"], errors="coerce").astype("Int64")
df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
df = df.dropna(subset=["sp_id", "native_state"]).copy()

rng = np.random.default_rng(42)
speaker_split = {}
state_summary = []

# Assign whole speakers to splits within each state.
for state, group in df.groupby("native_state"):
    speakers = np.array(sorted(group["sp_id"].unique()))
    rng.shuffle(speakers)
    n = len(speakers)

    # Reserve test speakers only where a state has enough speakers.
    n_test = min(max(1, round(0.20 * n)), n - 2) if n >= 3 else 0
    remaining = n - n_test
    n_val = min(max(1, round(0.10 * n)), remaining - 2) if n >= 4 else 0

    test_ids = speakers[:n_test]
    val_ids = speakers[n_test:n_test + n_val]
    train_ids = speakers[n_test + n_val:]

    for sid in train_ids:
        speaker_split[int(sid)] = "train"
    for sid in val_ids:
        speaker_split[int(sid)] = "validation"
    for sid in test_ids:
        speaker_split[int(sid)] = "test"

    state_summary.append({
        "state": state,
        "all_speakers": n,
        "train_speakers": len(train_ids),
        "val_speakers": len(val_ids),
        "test_speakers": len(test_ids),
    })

df["split"] = df["sp_id"].map(lambda sid: speaker_split[int(sid)])

print("\n=== SIMULATED SPEAKER SPLIT BY STATE ===")
print(pd.DataFrame(state_summary).to_string(index=False))

print("\n=== OVERALL DATA PER SPLIT ===")
overall = df.groupby("split").agg(
    recordings=("record_id", "size"),
    speakers=("sp_id", "nunique"),
    hours=("duration", lambda x: round(x.sum() / 3600, 2)),
)
print(overall.to_string())

print("\n=== WEST BENGAL DATA PER SPLIT ===")
wb = df[df["native_state"] == "West Bengal"]
print(wb.groupby("split").agg(
    recordings=("record_id", "size"),
    speakers=("sp_id", "nunique"),
    hours=("duration", lambda x: round(x.sum() / 3600, 2)),
).to_string())

print("\n=== SPEAKER LEAKAGE CHECK ===")
speaker_sets = {
    split: set(group["sp_id"].dropna().astype(int))
    for split, group in df.groupby("split")
}
for a, b in [("train", "validation"), ("train", "test"),
             ("validation", "test")]:
    overlap = speaker_sets[a] & speaker_sets[b]
    print(f"{a} vs {b}: {len(overlap)} overlapping speakers")

print("\n=== NON-TARGET TEST DATA BY STATE ===")
non_target = df[
    (df["split"] == "test") &
    (df["native_state"] != "West Bengal")
]
print(non_target.groupby("native_state").agg(
    recordings=("record_id", "size"),
    speakers=("sp_id", "nunique"),
    hours=("duration", lambda x: round(x.sum() / 3600, 2)),
).to_string())
