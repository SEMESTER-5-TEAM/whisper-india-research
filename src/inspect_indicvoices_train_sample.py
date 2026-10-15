
from collections import Counter
from itertools import islice

from datasets import load_dataset, Audio

print("Loading first 500 IndicVoices Hindi TRAIN records...")

ds = load_dataset(
    "ai4bharat/IndicVoices",
    "hindi",
    split="train",
    streaming=True,
    token=True,
)

ds = ds.cast_column("audio_filepath", Audio(decode=False))

rows = list(islice(ds, 500))

print("\nRows inspected:", len(rows))
print("\nState values and counts:")

counts = Counter(str(row.get("state")) for row in rows)

for state, count in counts.most_common():
    print(f"{state}: {count}")

print("\nFirst 10 records:")
for row in rows[:10]:
    print({
        "state": row.get("state"),
        "speaker_id": row.get("speaker_id"),
        "duration": row.get("duration"),
    })
