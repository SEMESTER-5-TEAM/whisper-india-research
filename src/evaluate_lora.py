import argparse, gc, io, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL_NAME = "openai/whisper-small"
META = Path("data/metadata/lahaja_speaker_disjoint_split.csv")
ADAPTER_DIR = Path("models/lora_generic_hindi/adapter_epoch_1")
OUT_DIR = Path("results/lora_generic_hindi")

def resolve_shard(value):
    value = str(value).replace("\\", "/")
    return Path(value) if value.startswith("data/") else Path("data/raw/lahaja/data") / value

def norm(s):
    s = str(s).lower().strip()
    return " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in s).split())

def edit_distance(a, b):
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j-1] + (ca != cb)))
        prev = cur
    return prev[-1]

def metric_counts(ref, hyp):
    rw, hw = norm(ref).split(), norm(hyp).split()
    rc, hc = norm(ref).replace(" ", ""), norm(hyp).replace(" ", "")
    return edit_distance(rw, hw), len(rw), edit_distance(rc, hc), len(rc)

class AudioReader:
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)
        self.path = None
        self.shard = None
    def __len__(self): return len(self.frame)
    def get(self, i):
        row = self.frame.iloc[i]
        path = resolve_shard(row["source_shard"])
        if path != self.path:
            self.shard, self.path = pd.read_parquet(path), path
        obj = self.shard.iloc[int(row["source_row"])]["audio_filepath"]
        audio, sr = sf.read(io.BytesIO(obj["bytes"]), dtype="float32")
        if audio.ndim > 1: audio = audio.mean(axis=1)
        if sr != 16000: raise ValueError(f"Expected 16 kHz, got {sr}: {path}")
        return audio[:30*sr], str(row["verbatim"])

def transcribe(frame, processor, model, device, label):
    reader = AudioReader(frame)
    rows, started = [], time.time()
    model.eval()
    for i in range(len(reader)):
        audio, ref = reader.get(i)
        feats = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype=torch.float32)
        t0 = time.time()
        with torch.inference_mode():
            ids = model.generate(input_features=feats, language="hindi", task="transcribe",
                                 do_sample=False, max_new_tokens=225)
        hyp = processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
        row = frame.iloc[i]
        rows.append({"record_id": row.get("record_id", i), "source_shard": row["source_shard"],
                     "source_row": int(row["source_row"]), "sp_id": row["sp_id"],
                     "state": row.get("native_state", ""), "district": row.get("native_district", ""),
                     "reference": ref, "prediction": hyp, "processing_seconds": round(time.time()-t0, 4),
                     "status": "success"})
        if i == 0 or (i+1) % 25 == 0 or i+1 == len(reader):
            print(f"{label}: {i+1}/{len(reader)} | elapsed={time.time()-started:.1f}s", flush=True)
        del feats, ids
    return pd.DataFrame(rows)

def summarize(df, name):
    wn = wd = cn = cd = 0
    perw, perc = [], []
    for r in df.itertuples():
        a, b, c, d = metric_counts(r.reference, r.prediction)
        wn += a; wd += b; cn += c; cd += d
        perw.append(a/max(b, 1)); perc.append(c/max(d, 1))
    return {"model": name, "n_recordings": len(df), "corpus_wer": wn/max(wd,1),
            "corpus_cer": cn/max(cd,1), "mean_recording_wer": float(np.mean(perw)),
            "mean_recording_cer": float(np.mean(perc)), "reference_words": wd,
            "reference_characters_no_spaces": cd,
            "mean_seconds_per_recording": float(df.processing_seconds.mean())}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_samples", type=int, default=0, help="0 evaluates all held-out test recordings")
    a = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; activate whisper-india.")
    if not ADAPTER_DIR.exists(): raise FileNotFoundError(f"Adapter not found: {ADAPTER_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(META)
    test = meta[meta.split.eq("test")].copy().reset_index(drop=True)
    if test.empty: raise ValueError("No rows marked split=test.")
    if a.max_samples: test = test.head(a.max_samples).copy().reset_index(drop=True)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Held-out test recordings:", len(test), "| speakers:", test.sp_id.nunique())
    print("Both systems use Hindi transcription and greedy decoding.")
    device = torch.device("cuda")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    print("\nLoading original Whisper-small...")
    base = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, low_cpu_mem_usage=True).to(device)
    base.generation_config.language = "hindi"; base.generation_config.task = "transcribe"
    base_df = transcribe(test, processor, base, device, "baseline")
    base_df.to_csv(OUT_DIR/"whisper_small_test_predictions.csv", index=False, encoding="utf-8-sig")
    bm = summarize(base_df, "original_whisper_small")
    del base, base_df; gc.collect(); torch.cuda.empty_cache()
    print("\nLoading Whisper-small + generic Hindi LoRA...")
    base = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    adapted = PeftModel.from_pretrained(base, str(ADAPTER_DIR)).to(device)
    adapted.generation_config.language = "hindi"; adapted.generation_config.task = "transcribe"
    lora_df = transcribe(test, processor, adapted, device, "lora")
    lora_df.to_csv(OUT_DIR/"whisper_small_lora_test_predictions.csv", index=False, encoding="utf-8-sig")
    lm = summarize(lora_df, "generic_hindi_lora")
    del adapted, base, lora_df; gc.collect(); torch.cuda.empty_cache()
    result = {"evaluation_note": "Same held-out LAHAJA test recordings and Hindi greedy decoding for both models. WER/CER use lowercase Unicode alphanumeric normalization, punctuation removal, whitespace normalization; CER excludes spaces.",
              "baseline": bm, "lora": lm,
              "delta_lora_minus_baseline": {"corpus_wer": lm["corpus_wer"]-bm["corpus_wer"],
                                            "corpus_cer": lm["corpus_cer"]-bm["corpus_cer"]}}
    (OUT_DIR/"metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([bm,lm]).to_csv(OUT_DIR/"metrics.csv", index=False, encoding="utf-8-sig")
    print("\nFINAL METRICS\n", json.dumps(result, indent=2, ensure_ascii=False))
    print("Saved outputs in:", OUT_DIR.resolve())

if __name__ == "__main__": main()
