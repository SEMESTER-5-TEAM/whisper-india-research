
import gc
import io
import time

import pandas as pd
import soundfile as sf
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model

MODEL_NAME = "openai/whisper-small"
METADATA_PATH = "data/metadata/lahaja_speaker_disjoint_split.csv"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable.")

device = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0))
print("Free VRAM before loading:", round(
    torch.cuda.mem_get_info(0)[0] / 1024**3, 2), "GB")

df = pd.read_csv(METADATA_PATH)
row = df[df["split"] == "train"].iloc[0]

shard_path = str(row["source_shard"])
if not shard_path.startswith("data/"):
    shard_path = "data/raw/lahaja/data/" + shard_path

print("Reading one training recording...")
shard = pd.read_parquet(shard_path)
audio_object = shard.iloc[int(row["source_row"])]["audio_filepath"]

audio, sample_rate = sf.read(
    io.BytesIO(audio_object["bytes"]), dtype="float32"
)
if sample_rate != 16000:
    raise ValueError(f"Expected 16000 Hz, got {sample_rate}")

del shard
gc.collect()

processor = WhisperProcessor.from_pretrained(MODEL_NAME)
features = processor.feature_extractor(
    audio, sampling_rate=sample_rate, return_tensors="pt"
).input_features.to(device=device, dtype=torch.float32)

labels = processor.tokenizer(
    str(row["verbatim"]), return_tensors="pt"
).input_ids.to(device)

print("Input features:", tuple(features.shape))
print("Label shape:", tuple(labels.shape))
print("Loading model in FP32...")

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
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-5
)

print("Starting one forward/backward pass...")
start = time.time()
optimizer.zero_grad(set_to_none=True)

with torch.autocast(device_type="cuda", enabled=False):
    output = model(input_features=features, labels=labels)
    loss = output.loss

if not torch.isfinite(loss):
    raise RuntimeError(f"Non-finite loss: {loss.item()}")

loss.backward()
torch.nn.utils.clip_grad_norm_(
    [p for p in model.parameters() if p.requires_grad], 1.0
)
optimizer.step()

torch.cuda.synchronize()
elapsed = time.time() - start

print("Pilot passed.")
print("Loss:", round(loss.item(), 4))
print("Forward/backward/optimizer time:", round(elapsed, 2), "seconds")
print("Peak allocated VRAM:", round(
    torch.cuda.max_memory_allocated() / 1024**3, 2), "GB")
print("Peak reserved VRAM:", round(
    torch.cuda.max_memory_reserved() / 1024**3, 2), "GB")

del model, optimizer, output, features, labels
gc.collect()
torch.cuda.empty_cache()
print("GPU pilot finished.")
