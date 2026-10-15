#!/usr/bin/env python
"""Score the four Whisper-small conditions on the same LAHAJA test recordings."""
from pathlib import Path
import json
import re
import unicodedata
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "lora_generic_hindi"
CONDITIONS = {
    "baseline_forced_hindi": RESULTS / "whisper_small_test_predictions.csv",
    "lora_forced_hindi": RESULTS / "whisper_small_lora_test_predictions.csv",
    "baseline_default": RESULTS / "whisper_small_default_test_predictions.csv",
    "lora_default": RESULTS / "whisper_small_lora_default_test_predictions.csv",
}
KEY_COLUMNS = ["source_shard", "source_row"]


def normalize_text(value):
    if pd.isna(value):
        value = ""
    value = unicodedata.normalize("NFKC", str(value)).lower()
    chars = []
    for ch in value:
        category = unicodedata.category(ch)
        chars.append(ch if category[0] in ("L", "N") or ch.isspace() else " ")
    return re.sub(r"\s+", " ", "".join(chars)).strip()


def edit_counts(ref, hyp):
    """Levenshtein alignment with substitution/deletion/insertion counts."""
    n, m = len(ref), len(hyp)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0], back[i][0] = i, "D"
    for j in range(1, m + 1):
        cost[0][j], back[0][j] = j, "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = ref[i - 1] == hyp[j - 1]
            options = (
                (cost[i - 1][j - 1] + (0 if same else 1), "C" if same else "S"),
                (cost[i - 1][j] + 1, "D"),
                (cost[i][j - 1] + 1, "I"),
            )
            cost[i][j], back[i][j] = min(options, key=lambda x: x[0])
    counts = {"S": 0, "D": 0, "I": 0}
    i, j = n, m
    while i or j:
        op = back[i][j]
        if op == "C":
            i -= 1
            j -= 1
        elif op == "S":
            counts["S"] += 1
            i -= 1
            j -= 1
        elif op == "D":
            counts["D"] += 1
            i -= 1
        elif op == "I":
            counts["I"] += 1
            j -= 1
        else:
            raise RuntimeError("Invalid alignment traceback")
    return counts


def load_condition(name, path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {name} prediction file: {path}")
    df = pd.read_csv(path, keep_default_na=False)
    required = set(KEY_COLUMNS + ["reference", "prediction", "status"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing required columns: {sorted(missing)}")
    if df[KEY_COLUMNS].duplicated().any():
        raise ValueError(f"{name}: duplicate source_shard/source_row keys")
    for col in KEY_COLUMNS:
        df[col] = df[col].astype(str)
    df["reference_norm"] = df["reference"].map(normalize_text)
    df["prediction_norm"] = df["prediction"].map(normalize_text)
    return df


def main():
    frames = {name: load_condition(name, path) for name, path in CONDITIONS.items()}
    first = frames[next(iter(frames))]
    keys = set(map(tuple, first[KEY_COLUMNS].to_numpy()))
    refs = dict(zip(map(tuple, first[KEY_COLUMNS].to_numpy()), first["reference_norm"]))
    for name, df in frames.items():
        other_keys = set(map(tuple, df[KEY_COLUMNS].to_numpy()))
        if other_keys != keys:
            raise ValueError(f"{name}: key mismatch; missing={len(keys-other_keys)}, extra={len(other_keys-keys)}")
        for key, ref in zip(map(tuple, df[KEY_COLUMNS].to_numpy()), df["reference_norm"]):
            if ref != refs[key]:
                raise ValueError(f"{name}: reference mismatch for {key}")

    summaries, details, statuses = [], [], []
    for name, df in frames.items():
        sums = {"S": 0, "D": 0, "I": 0}
        total_ref_words = total_ref_chars = total_char_errors = 0
        empty_predictions = int((df["prediction_norm"] == "").sum())
        per_record = []
        for row in df.itertuples(index=False):
            rw = row.reference_norm.split() if row.reference_norm else []
            hw = row.prediction_norm.split() if row.prediction_norm else []
            wc = edit_counts(rw, hw)
            rc = row.reference_norm.replace(" ", "")
            hc = row.prediction_norm.replace(" ", "")
            cc = edit_counts(list(rc), list(hc))
            for k in sums:
                sums[k] += wc[k]
            total_ref_words += len(rw)
            total_ref_chars += len(rc)
            char_errors = sum(cc.values())
            total_char_errors += char_errors
            per_record.append({
                "source_shard": row.source_shard, "source_row": row.source_row,
                "sp_id": getattr(row, "sp_id", ""), "reference": row.reference,
                "prediction": row.prediction, "reference_normalized": row.reference_norm,
                "prediction_normalized": row.prediction_norm, "reference_words": len(rw),
                "S": wc["S"], "D": wc["D"], "I": wc["I"],
                "word_errors": sum(wc.values()),
                "record_WER": sum(wc.values()) / len(rw) if rw else None,
                "reference_characters_excluding_spaces": len(rc),
                "character_errors": char_errors,
                "record_CER": char_errors / len(rc) if rc else None,
                "status": row.status,
            })
        errors = sum(sums.values())
        status_counts = df["status"].astype(str).value_counts(dropna=False).to_dict()
        good = {"ok", "success", "completed"}
        bad_rows = int((~df["status"].astype(str).str.lower().isin(good)).sum())
        summaries.append({
            "condition": name, "recordings": len(df),
            "non_success_status_rows": bad_rows, "empty_normalized_predictions": empty_predictions,
            "reference_words": total_ref_words,
            "reference_characters_excluding_spaces": total_ref_chars,
            "substitutions": sums["S"], "deletions": sums["D"], "insertions": sums["I"],
            "word_errors": errors,
            "corpus_WER": errors / total_ref_words if total_ref_words else None,
            "character_errors": total_char_errors,
            "corpus_CER_no_spaces": total_char_errors / total_ref_chars if total_ref_chars else None,
        })
        for item in per_record:
            item["condition"] = name
            details.append(item)
        statuses.append({"condition": name, "total_rows": len(df),
                         "status_counts": status_counts,
                         "empty_normalized_predictions": empty_predictions})

    metrics = pd.DataFrame(summaries)
    metrics_path = RESULTS / "four_condition_metrics.csv"
    details_path = RESULTS / "four_condition_per_record.csv"
    status_path = RESULTS / "four_condition_status_summary.json"
    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame(details).to_csv(details_path, index=False)
    with status_path.open("w", encoding="utf-8") as f:
        json.dump(statuses, f, ensure_ascii=False, indent=2)

    print("\nFOUR-CONDITION CORPUS METRICS")
    cols = ["condition", "recordings", "non_success_status_rows", "reference_words",
            "corpus_WER", "corpus_CER_no_spaces", "substitutions", "deletions", "insertions"]
    print(metrics[cols].to_string(index=False, formatters={
        "corpus_WER": lambda x: f"{x:.4f}",
        "corpus_CER_no_spaces": lambda x: f"{x:.4f}",
    }))
    print("\nSaved:")
    print(metrics_path.relative_to(ROOT))
    print(details_path.relative_to(ROOT))
    print(status_path.relative_to(ROOT))
    if (metrics["non_success_status_rows"] > 0).any():
        print("\nWARNING: Non-success status rows exist. Inspect the status summary before interpreting scores.")
    if (metrics["empty_normalized_predictions"] > 0).any():
        print("\nWARNING: Empty normalized predictions exist. Inspect per-record results.")


if __name__ == "__main__":
    main()
