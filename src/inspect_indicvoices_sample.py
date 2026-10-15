
from collections import Counter
from itertools import islice

from datasets import load_dataset, Audio

print("Loading a small sample of IndicVoices Hindi validation data...")

ds = load_dataset(
    "ai4bharat/IndicVoices",
    "hindi",
    split="valid",
    streaming=True,
    token=True,
)
ds = ds.cast_column("audio_filepath", Audio(decode=False))

rows = list(islice(ds, 500))

print("\nRows inspected:", len(rows))
print("\nState values and counts in this sample:")

counts = Counter(repr(row.get("state")) for row in rows)

for state, count in counts.most_common():
    print(f"{state}: {count}")

print("\nFirst 10 state values:")
for row in rows[:10]:
    print(repr(row.get("state")))
