
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results" / "forced_language"
META = ROOT / "data" / "metadata" / "evaluation_subset_60_per_state.csv"

forced = pd.read_csv(BASE / "whisper_small_forced_language.csv")
meta = pd.read_csv(META)

print("FORCED-LANGUAGE RESULT COUNTS")
print(forced.groupby("dataset").size().to_string())

print("\nMETADATA SPEAKER-ID AVAILABILITY")
print("Total metadata rows:", len(meta))
print("Non-missing sp_id:", int(meta["sp_id"].notna().sum()))
print("Distinct non-missing sp_id:", meta["sp_id"].nunique(dropna=True))

merged = forced.merge(
    meta[["eval_id", "dataset", "record_id", "sp_id"]],
    on=["eval_id", "dataset", "record_id"],
    how="left",
    validate="one_to_one",
)

print("\nMATCHED FORCED-LANGUAGE SPEAKERS")
for dataset, part in merged.groupby("dataset"):
    print(
        f"{dataset}: rows={len(part)}, "
        f"missing IDs={part['sp_id'].isna().sum()}, "
        f"unique speakers={part['sp_id'].nunique(dropna=True)}"
    )
