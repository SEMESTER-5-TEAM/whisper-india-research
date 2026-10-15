
from itertools import islice
from collections import Counter
from datasets import load_dataset

print("Opening IndicVoices Hindi training split...")

ds = load_dataset(
    "ai4bharat/IndicVoices",
    "hindi",
    split="train",
    streaming=True,
    token=True,
)

# Exclude the audio column before reading records.
ds = ds.select_columns([
    "state",
    "speaker_id",
    "duration",
    "lang",
])

print("Selected columns:", ds.column_names)
print("Reading up to 500 metadata-only records...")

rows = list(islice(ds, 500))

print("Records read:", len(rows))
print("\nState counts in this sample:")

counts = Counter(row.get("state") for row in rows)

for state, count in counts.most_common():
    print(f"{state}: {count}")

print("\nFirst record:")
if rows:
    print(rows[0])
