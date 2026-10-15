
import gc
import io
import os
import time

import pandas as pd
import soundfile as sf
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model

MODEL_NAME = "openai/whisper-small"
METADATA_PATH = "data/metadata/lahaja_speaker_disjoint_split.csv"
N = 10

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable.")

device = torch.device("cuda")
torch.cuda.reset_peak_memory_stats()

df = pd.read_csv(METADATA_PATH)
train = df[df["split"] == "train"].copy()

# Use ten different training recordings from one Parquet shard.
first_shard = os.path.basename(str(train.iloc[0]["source_shard"]))
train["shard_name"] = train["source_shard"].map(
    lambda x: os.path.basename(str(x))
)
rows = train[train["shard_name"] == first_shard].head(N)

if len(rows) < N:
    raise RuntimeError(f"Only {len(rows)} examples found in this shard.")

shard_path = os.path.join("data", "raw", "lahaja", "data", first_shard)
print("Reading shard:", shard_path)
shard = pd.read_parquet(shard_path)

processor = WhisperProcessor.from_pretrained(MODEL_NAME)
examples = []

for _, row in rows.iterrows():
    obj = shard.iloc[int(row["source_row"])]["audio_filepath"]
    audio, sr = sf.read(io.BytesIO(obj["bytes"]), dtype="float32")

    if sr != 16000:
        raise ValueError(f"Unexpected sample rate: {sr}")

    features = processor.feature_extractor(
        audio, sampling_rate=sr, return_tensors="pt"
    ).input_features

    labels = processor.tokenizer(
        str(row["verbatim"]), return_tensors="pt"
    ).input_ids

    examples.append((features, labels))

del shard
gc.collect()

print("Prepared distinct recordings:", len(examples))
print("Loading Whisper-small in FP32...")

model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_NAME, dtype=torch.float32, low_cpu_mem_usage=True
)
model.config.use_cache = False
model.generation_config.language = "hindi"
model.generation_config.task = "transcribe"

model = get_peft_model(
    model,
    LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )
)
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
model.enable_input_require_grads()
model.to(device)
model.train()

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=1e-5
)

times = []
losses = []

for i, (features, labels) in enumerate(examples, start=1):
    features = features.to(device=device, dtype=torch.float32)
    labels = labels.to(device)

    optimizer.zero_grad(set_to_none=True)
    start = time.time()

    with torch.autocast(device_type="cuda", enabled=False):
        output = model(input_features=features, labels=labels)
        loss = output.loss

    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss at step {i}: {loss.item()}")

    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0
    )
    optimizer.step()
    torch.cuda.synchronize()

    elapsed = time.time() - start
    times.append(elapsed)
    losses.append(loss.item())
    print(f"Step {i}/{N}: loss={loss.item():.4f}, time={elapsed:.2f}s")

print("\nBenchmark complete.")
print("Mean step time:", round(sum(times) / len(times), 2), "seconds")
print("Total timed training:", round(sum(times), 2), "seconds")
print("Peak allocated VRAM:", round(
    torch.cuda.max_memory_allocated() / 1024**3, 2), "GB")
print("Peak reserved VRAM:", round(
    torch.cuda.max_memory_reserved() / 1024**3, 2), "GB")

del model, optimizer, examples
gc.collect()
torch.cuda.empty_cache()
