import argparse, gc, io, json, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from peft import LoraConfig, get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL_NAME = "openai/whisper-small"
META = Path("data/metadata/lahaja_speaker_disjoint_split.csv")
OUT = Path("models/lora_generic_hindi")


def shard_path(value):
    value = str(value).replace("\\", "/")
    return Path(value) if value.startswith("data/") else Path("data/raw/lahaja/data") / value


class LahajaDataset(torch.utils.data.Dataset):
    def __init__(self, frame, processor, max_seconds=30):
        # Keep rows grouped by shard so Parquet files are not reopened for every recording.
        self.frame = frame.sort_values(["source_shard", "source_row"]).reset_index(drop=True)
        self.processor = processor
        self.max_samples = int(max_seconds * 16000)
        self.path = None
        self.shard = None

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        path = shard_path(row.source_shard)
        if path != self.path:
            self.shard = pd.read_parquet(path)
            self.path = path
        obj = self.shard.iloc[int(row.source_row)]["audio_filepath"]
        audio, sr = sf.read(io.BytesIO(obj["bytes"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            raise ValueError(f"Expected 16 kHz audio, got {sr}: {path}")
        audio = audio[:self.max_samples]
        text = str(row.verbatim).strip()
        if not text or text.lower() == "nan":
            raise ValueError(f"Empty transcript at metadata row {i}")
        feats = self.processor.feature_extractor(
            audio, sampling_rate=sr, return_tensors="pt"
        ).input_features[0]
        labels = self.processor.tokenizer(
            text, truncation=True, max_length=448, return_tensors="pt"
        ).input_ids[0]
        return {"input_features": feats, "labels": labels}


def collate(items):
    x = torch.stack([v["input_features"] for v in items])
    n = max(len(v["labels"]) for v in items)
    y = torch.full((len(items), n), -100, dtype=torch.long)
    for i, v in enumerate(items):
        y[i, :len(v["labels"])] = v["labels"]
    return {"input_features": x, "labels": y}


def epoch(model, loader, device, optimizer=None, accumulation=8):
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    losses, start = [], time.time()
    for step, batch in enumerate(loader, 1):
        x = batch["input_features"].to(device, dtype=torch.float32)
        y = batch["labels"].to(device)
        with torch.set_grad_enabled(training):
            loss = model(input_features=x, labels=y).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at batch {step}: {loss.item()}")
        losses.append(loss.detach().item())
        if training:
            (loss / accumulation).backward()
            if step % accumulation == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 25 == 0 or step == len(loader):
            print(f"{'train' if training else 'valid'} {step}/{len(loader)} "
                  f"loss={losses[-1]:.4f} elapsed={time.time()-start:.1f}s",
                  flush=True)
        del x, y, loss
    return {"mean_loss": float(np.mean(losses)), "batches": len(losses),
            "seconds": round(time.time()-start, 2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_val_samples", type=int, default=0)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--max_audio_seconds", type=float, default=30)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; activate whisper-india.")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    df = pd.read_csv(META)
    required = {"split", "source_shard", "source_row", "verbatim", "sp_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {sorted(required-set(df.columns))}")
    tr = df[df.split.eq("train")].copy()
    va = df[df.split.eq("validation")].copy()
    te = df[df.split.eq("test")].copy()
    if tr.empty or va.empty or te.empty:
        raise ValueError(f"Unexpected split counts: {df.split.value_counts().to_dict()}")
    if set(tr.sp_id.astype(str)) & set(va.sp_id.astype(str)):
        raise ValueError("Speaker leakage between train and validation.")
    if set(tr.sp_id.astype(str)) & set(te.sp_id.astype(str)):
        raise ValueError("Speaker leakage between train and test.")
    if set(va.sp_id.astype(str)) & set(te.sp_id.astype(str)):
        raise ValueError("Speaker leakage between validation and test.")
    if a.max_train_samples:
        tr = tr.sample(min(a.max_train_samples, len(tr)), random_state=a.seed)
    if a.max_val_samples:
        va = va.sample(min(a.max_val_samples, len(va)), random_state=a.seed)
    print(f"Train: {len(tr)} recordings / {tr.sp_id.nunique()} speakers")
    print(f"Validation: {len(va)} recordings / {va.sp_id.nunique()} speakers")
    print(f"Test reserved: {len(te)} recordings / {te.sp_id.nunique()} speakers")
    OUT.mkdir(parents=True, exist_ok=True)
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.config.use_cache = False
    model.generation_config.language = "hindi"
    model.generation_config.task = "transcribe"
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none"
    ))
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.to(device)
    model.print_trainable_parameters()
    train_ds = LahajaDataset(tr, processor, a.max_audio_seconds)
    val_ds = LahajaDataset(va, processor, a.max_audio_seconds)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate
    )
    opt = torch.optim.AdamW(
        [q for q in model.parameters() if q.requires_grad], lr=a.learning_rate
    )
    cfg = vars(a) | {"model": MODEL_NAME, "dtype": "float32",
                     "train_recordings": len(tr), "validation_recordings": len(va),
                     "train_speakers": int(tr.sp_id.nunique()),
                     "validation_speakers": int(va.sp_id.nunique())}
    (OUT / "run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    history = []
    for e in range(1, a.epochs + 1):
        print(f"\nEpoch {e}/{a.epochs}", flush=True)
        train_stats = epoch(model, train_loader, device, opt,
                            max(1, a.gradient_accumulation_steps))
        val_stats = epoch(model, val_loader, device)
        rec = {"epoch": e, "train": train_stats, "validation": val_stats}
        history.append(rec)
        (OUT / "training_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        save = OUT / f"adapter_epoch_{e}"
        model.save_pretrained(save)
        processor.save_pretrained(save)
        print(json.dumps(rec, indent=2), flush=True)
        gc.collect(); torch.cuda.empty_cache()
    print("Training complete. Adapter saved under", OUT.resolve())


if __name__ == "__main__":
    main()
