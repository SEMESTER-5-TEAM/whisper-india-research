
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

files = [
    ROOT / "data" / "metadata" / "evaluation_subset_60_per_state.csv",
    ROOT / "data" / "metadata" / "lahaja_metadata.csv",
    ROOT / "data" / "metadata" / "svarah_metadata.csv",
]

for path in files:
    print("\n" + "=" * 70)
    print("FILE:", path.name)

    df = pd.read_csv(path, nrows=5)
    print("COLUMNS:")
    for col in df.columns:
        print(" ", repr(col))

    speaker_cols = [
        col for col in df.columns
        if "speaker" in col.lower()
        or "participant" in col.lower()
    ]

    print("POSSIBLE SPEAKER COLUMNS:", speaker_cols)
