"""Fast regional and speaker-bootstrap analysis of saved Whisper/LoRA predictions.
Run from project root: python src\analyze_lora_results.py
Does not load models or rerun inference.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("results/lora_generic_hindi")
BASE = OUT / "whisper_small_test_predictions.csv"
LORA = OUT / "whisper_small_lora_test_predictions.csv"
N_BOOT = 5000
SEED = 42


def norm(s):
    s = str(s).lower().strip()
    return " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in s).split())


def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def error_counts(ref, hyp):
    rw, hw = norm(ref).split(), norm(hyp).split()
    rc, hc = norm(ref).replace(" ", ""), norm(hyp).replace(" ", "")
    return edit_distance(rw, hw), len(rw), edit_distance(rc, hc), len(rc)


def attach_counts(df, pred_col, suffix):
    vals = [error_counts(r, h) for r, h in zip(df["reference"], df[pred_col])]
    out = df[["source_shard", "source_row", "sp_id", "state", "district"]].copy()
    out[[f"we_{suffix}", f"rw_{suffix}", f"ce_{suffix}", f"rc_{suffix}"]] = pd.DataFrame(vals, index=out.index)
    return out


def metrics_from_sums(we, rw, ce, rc):
    return {"corpus_wer": float(we / max(rw, 1)),
            "corpus_cer": float(ce / max(rc, 1)),
            "word_errors": int(we), "reference_words": int(rw),
            "character_errors": int(ce), "reference_characters_no_spaces": int(rc)}


def main():
    for path in (BASE, LORA):
        if not path.exists():
            raise FileNotFoundError(f"Missing saved predictions: {path}")
    b, l = pd.read_csv(BASE), pd.read_csv(LORA)
    keys = ["source_shard", "source_row"]
    m = b.merge(l, on=keys, how="inner", suffixes=("_base", "_lora"), validate="one_to_one")
    if len(m) != len(b) or len(m) != len(l):
        raise ValueError(f"Prediction alignment mismatch: baseline={len(b)}, LoRA={len(l)}, matched={len(m)}")
    if not (m.reference_base.astype(str).values == m.reference_lora.astype(str).values).all():
        raise ValueError("References differ between baseline and LoRA.")
    if not (m.sp_id_base.astype(str).values == m.sp_id_lora.astype(str).values).all():
        raise ValueError("Speaker IDs differ between baseline and LoRA.")

    common = pd.DataFrame({
        "source_shard": m.source_shard,
        "source_row": m.source_row,
        "sp_id": m.sp_id_base.astype(str),
        "state": m.state_base.fillna("Unknown").astype(str),
        "district": m.district_base.fillna("Unknown").astype(str),
        "reference": m.reference_base.astype(str),
        "prediction_base": m.prediction_base.fillna("").astype(str),
        "prediction_lora": m.prediction_lora.fillna("").astype(str),
    }).reset_index(drop=True)

    print(f"Computing per-recording errors once for {len(common)} paired recordings...", flush=True)
    base_counts = attach_counts(common, "prediction_base", "base")
    lora_counts = attach_counts(common, "prediction_lora", "lora")
    d = pd.concat([common[["sp_id", "state", "district"]], base_counts.drop(columns=["source_shard", "source_row", "sp_id", "state", "district"]),
                   lora_counts.drop(columns=["source_shard", "source_row", "sp_id", "state", "district"])], axis=1)

    def full_metrics(suffix):
        return metrics_from_sums(d[f"we_{suffix}"].sum(), d[f"rw_{suffix}"].sum(),
                                 d[f"ce_{suffix}"].sum(), d[f"rc_{suffix}"].sum())

    bm, lm = full_metrics("base"), full_metrics("lora")
    print("FULL TEST SET")
    print("Baseline:", {"recordings": len(d), "speakers": d.sp_id.nunique(), **bm})
    print("LoRA:    ", {"recordings": len(d), "speakers": d.sp_id.nunique(), **lm})

    rows = []
    for level, group in d.groupby("state", dropna=False):
        pass
    for level in ("state", "district"):
        for region, g in d.groupby(level, dropna=False):
            x = metrics_from_sums(g.we_base.sum(), g.rw_base.sum(), g.ce_base.sum(), g.rc_base.sum())
            y = metrics_from_sums(g.we_lora.sum(), g.rw_lora.sum(), g.ce_lora.sum(), g.rc_lora.sum())
            rows.append({
                "level": level, "region": region, "recordings": int(len(g)),
                "speakers": int(g.sp_id.nunique()),
                "baseline_wer": x["corpus_wer"], "lora_wer": y["corpus_wer"],
                "wer_change_lora_minus_baseline": y["corpus_wer"] - x["corpus_wer"],
                "baseline_cer": x["corpus_cer"], "lora_cer": y["corpus_cer"],
                "cer_change_lora_minus_baseline": y["corpus_cer"] - x["corpus_cer"],
            })
    regional_path = OUT / "regional_comparison.csv"
    pd.DataFrame(rows).to_csv(regional_path, index=False, encoding="utf-8-sig")
    print("Saved regional table:", regional_path, flush=True)

    # Pre-aggregate counts by speaker. Bootstrap then sums tiny arrays; no repeated edit-distance calculations.
    cols = ["we_base", "rw_base", "ce_base", "rc_base", "we_lora", "rw_lora", "ce_lora", "rc_lora"]
    sp = d.groupby("sp_id")[cols].sum()
    ids = sp.index.to_numpy()
    arr = sp.to_numpy(dtype=np.int64)
    rng = np.random.default_rng(SEED)
    wer_diffs = np.empty(N_BOOT, dtype=float)
    cer_diffs = np.empty(N_BOOT, dtype=float)
    print(f"Bootstrapping {N_BOOT} speaker-level replicates across {len(ids)} speakers...", flush=True)
    for i in range(N_BOOT):
        ix = rng.integers(0, len(ids), size=len(ids))
        s = arr[ix].sum(axis=0)
        bwer = s[0] / max(s[1], 1)
        bcer = s[2] / max(s[3], 1)
        lwer = s[4] / max(s[5], 1)
        lcer = s[6] / max(s[7], 1)
        wer_diffs[i] = lwer - bwer
        cer_diffs[i] = lcer - bcer

    result = {
        "test_recordings": int(len(d)), "test_speakers": int(len(ids)),
        "bootstrap_replicates": N_BOOT,
        "bootstrap_unit": "speaker; paired baseline and LoRA predictions retained together",
        "seed": SEED, "baseline": bm, "lora": lm,
        "change_lora_minus_baseline": {
            "corpus_wer": lm["corpus_wer"] - bm["corpus_wer"],
            "corpus_cer": lm["corpus_cer"] - bm["corpus_cer"],
        },
        "speaker_bootstrap_95_percent_ci_for_change": {
            "corpus_wer": [float(np.quantile(wer_diffs, .025)), float(np.quantile(wer_diffs, .975))],
            "corpus_cer": [float(np.quantile(cer_diffs, .025)), float(np.quantile(cer_diffs, .975))],
        },
        "interpretation": "Negative changes favor LoRA. Region-level estimates with few speakers are exploratory."
    }
    boot_path = OUT / "speaker_bootstrap.json"
    boot_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nSPEAKER-AWARE PAIRED BOOTSTRAP")
    print(json.dumps(result, indent=2))
    print("Saved:", boot_path, flush=True)


if __name__ == "__main__":
    main()
