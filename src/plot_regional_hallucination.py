import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# FILES
# ============================================================

REVIEW_FILE = Path(
    "results/baseline/hallucination_review_expanded_completed.csv"
)

EVAL_FILE = Path(
    "data/metadata/evaluation_subset_60_per_state.csv"
)

OUTPUT_DIR = Path("results/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

review = pd.read_csv(REVIEW_FILE)
evaluation = pd.read_csv(EVAL_FILE)

print("Review records:", len(review))
print("Evaluation records:", len(evaluation))


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

review.columns = review.columns.str.strip()
evaluation.columns = evaluation.columns.str.strip()


# ============================================================
# CREATE COMMON STATE COLUMN
# ============================================================

# Svarah:
#   native_place_state
#
# LAHAJA:
#   native_state

if "state" not in evaluation.columns:

    evaluation["state"] = evaluation["native_place_state"].where(
        evaluation["native_place_state"].notna(),
        evaluation["native_state"]
    )

# The above may fail if one of the columns does not exist.
# Therefore explicitly handle the two datasets.

if "native_place_state" in evaluation.columns:

    if "native_state" in evaluation.columns:

        evaluation["state"] = evaluation[
            "native_place_state"
        ].fillna(
            evaluation["native_state"]
        )

    else:

        evaluation["state"] = evaluation[
            "native_place_state"
        ]

elif "native_state" in evaluation.columns:

    evaluation["state"] = evaluation[
        "native_state"
    ]

else:

    raise ValueError(
        "Could not find a state column or the expected "
        "native_place_state/native_state columns."
    )


# ============================================================
# NORMALIZE DATASET NAMES
# ============================================================

evaluation["dataset"] = (
    evaluation["dataset"]
    .astype(str)
    .str.lower()
    .str.strip()
)

review["dataset"] = (
    review["dataset"]
    .astype(str)
    .str.lower()
    .str.strip()
)

evaluation["state"] = (
    evaluation["state"]
    .astype(str)
    .str.strip()
)

review["state"] = (
    review["state"]
    .astype(str)
    .str.strip()
)


# ============================================================
# TOTAL RECORDS PER DATASET + STATE
# ============================================================

state_totals = (
    evaluation
    .groupby(["dataset", "state"])
    .size()
    .reset_index(name="total_records")
)


# ============================================================
# HUMAN-REVIEWED CANDIDATES
# ============================================================

candidates = (
    review
    .groupby(["dataset", "state"])
    .size()
    .reset_index(name="candidates_reviewed")
)


# ============================================================
# VALIDATED HALLUCINATIONS
# ============================================================

hallucinations = (
    review[
        review["human_label"]
        .astype(str)
        .str.upper()
        .str.strip()
        == "HALLUCINATION"
    ]
    .groupby(["dataset", "state"])
    .size()
    .reset_index(name="hallucinations")
)


# ============================================================
# MERGE
# ============================================================

regional = state_totals.merge(
    candidates,
    on=["dataset", "state"],
    how="left"
)

regional = regional.merge(
    hallucinations,
    on=["dataset", "state"],
    how="left"
)

regional["candidates_reviewed"] = (
    regional["candidates_reviewed"]
    .fillna(0)
    .astype(int)
)

regional["hallucinations"] = (
    regional["hallucinations"]
    .fillna(0)
    .astype(int)
)


# ============================================================
# CANDIDATE-BASED OBSERVED RATE
# ============================================================

regional["observed_hallucination_rate"] = (
    regional["hallucinations"]
    / regional["total_records"]
    * 100
)


# ============================================================
# LAHAJA ANALYSIS
# ============================================================

lahaja = regional[
    regional["dataset"] == "lahaja"
].copy()

lahaja = lahaja.sort_values(
    "observed_hallucination_rate",
    ascending=False
)


print("\n" + "=" * 75)
print("LAHAJA REGIONAL RESULTS")
print("=" * 75)

print(
    lahaja[
        [
            "state",
            "total_records",
            "candidates_reviewed",
            "hallucinations",
            "observed_hallucination_rate"
        ]
    ].to_string(index=False)
)


# ============================================================
# PRELIMINARY TOP 4
# ============================================================

TOP_N = 4

top_regions = lahaja.head(TOP_N)["state"].tolist()


print("\n" + "=" * 75)
print("PRELIMINARY TARGET REGIONS")
print("=" * 75)

for rank, state in enumerate(top_regions, start=1):

    row = lahaja[
        lahaja["state"] == state
    ].iloc[0]

    print(
        f"{rank}. {state}: "
        f"{int(row['hallucinations'])}/"
        f"{int(row['total_records'])} = "
        f"{row['observed_hallucination_rate']:.2f}%"
    )


# ============================================================
# GRAPH 1
# ALL LAHAJA REGIONS
# ============================================================

plt.figure(figsize=(14, 8))

bars = plt.bar(
    lahaja["state"],
    lahaja["observed_hallucination_rate"]
)

# Highlight top 4
for bar, state in zip(bars, lahaja["state"]):

    if state in top_regions:
        bar.set_alpha(1.0)
    else:
        bar.set_alpha(0.40)


# Mean line
mean_rate = lahaja[
    "observed_hallucination_rate"
].mean()

plt.axhline(
    mean_rate,
    linestyle="--",
    linewidth=1.5,
    label=f"Mean across states = {mean_rate:.2f}%"
)


plt.title(
    "Candidate-Based Observed Hallucination Rate by Region — LAHAJA"
)

plt.xlabel("Native state")

plt.ylabel(
    "Observed hallucination rate (%)"
)

plt.xticks(
    rotation=60,
    ha="right"
)

plt.legend()

plt.tight_layout()


graph1 = (
    OUTPUT_DIR /
    "lahaja_hallucination_rate_by_region.png"
)

plt.savefig(
    graph1,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nSaved:", graph1)


# ============================================================
# GRAPH 2
# CANDIDATES VS VALIDATED HALLUCINATIONS
# ============================================================

plt.figure(figsize=(14, 8))

x = range(len(lahaja))

plt.bar(
    x,
    lahaja["candidates_reviewed"],
    label="Reviewed candidates"
)

plt.bar(
    x,
    lahaja["hallucinations"],
    label="Validated hallucinations"
)

plt.xticks(
    list(x),
    lahaja["state"],
    rotation=60,
    ha="right"
)

plt.title(
    "Reviewed Candidates and Validated Hallucinations — LAHAJA"
)

plt.xlabel("Native state")

plt.ylabel("Number of recordings")

plt.legend()

plt.tight_layout()


graph2 = (
    OUTPUT_DIR /
    "lahaja_candidates_vs_hallucinations.png"
)

plt.savefig(
    graph2,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("Saved:", graph2)


# ============================================================
# GRAPH 3
# PRELIMINARY TARGET REGIONS
# ============================================================

selected = lahaja[
    lahaja["state"].isin(top_regions)
].copy()


# Preserve ranking order
selected = selected.sort_values(
    "observed_hallucination_rate",
    ascending=False
)


plt.figure(figsize=(9, 6))

bars = plt.bar(
    selected["state"],
    selected["observed_hallucination_rate"]
)


for bar, value in zip(
    bars,
    selected["observed_hallucination_rate"]
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{value:.2f}%",
        ha="center",
        va="bottom"
    )


plt.title(
    "Preliminary Target Regions for Region-Specific Adaptation"
)

plt.xlabel("Native state")

plt.ylabel(
    "Candidate-based observed hallucination rate (%)"
)

plt.tight_layout()


graph3 = (
    OUTPUT_DIR /
    "preliminary_target_regions.png"
)

plt.savefig(
    graph3,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("Saved:", graph3)


# ============================================================
# SAVE REGIONAL SUMMARY
# ============================================================

regional_output = (
    OUTPUT_DIR /
    "regional_hallucination_summary.csv"
)

regional.to_csv(
    regional_output,
    index=False
)

print("Saved:", regional_output)


# ============================================================
# FINAL METHODOLOGICAL NOTE
# ============================================================

print("\n" + "=" * 75)
print("IMPORTANT METHODOLOGICAL NOTE")
print("=" * 75)

print(
    "\nThe plotted rate is a "
    "candidate-based observed hallucination rate."
)

print(
    "It is NOT the complete hallucination prevalence "
    "of each state because only candidate recordings "
    "were manually reviewed."
)

print(
    "\nThe top regions are therefore "
    "PRELIMINARY TARGET REGIONS."
)

print(
    "\nFinal region selection must also consider "
    "IndicVoices Hindi adaptation-data availability."
)

print(
    "\nDo not interpret the regional differences "
    "as evidence that a state causes hallucinations."
)