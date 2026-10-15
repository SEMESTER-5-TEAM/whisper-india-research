
from datasets import load_dataset, Audio
from collections import Counter

print("Loading IndicVoices Hindi validation split...")

ds = load_dataset(
    "ai4bharat/IndicVoices",
    "hindi",
    split="valid",
    streaming=True,
    token=True,
)

ds = ds.cast_column("audio_filepath", Audio(decode=False))

state_counts = Counter()
total = 0

for row in ds:
    total += 1
    state = row.get("state")
    state_counts[str(state)] += 1

print("\nTotal records:", total)
print("\nActual state values in the dataset:")

for state, count in state_counts.most_common():
    print(f"{state!r}: {count}")

print("\nNumber of distinct state values:", len(state_counts))
