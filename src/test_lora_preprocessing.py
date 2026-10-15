
import io
import pandas as pd
import soundfile as sf
from transformers import WhisperProcessor

metadata_path = "data/metadata/lahaja_speaker_disjoint_split.csv"
df = pd.read_csv(metadata_path)
row = df[df["split"] == "train"].iloc[0]

shard_path = row["source_shard"]
if not shard_path.startswith("data/"):
    shard_path = "data/raw/lahaja/data/" + shard_path

shard = pd.read_parquet(shard_path)
audio_object = shard.iloc[int(row["source_row"])]["audio_filepath"]
audio, sample_rate = sf.read(
    io.BytesIO(audio_object["bytes"]),
    dtype="float32"
)

if sample_rate != 16000:
    raise ValueError(f"Expected 16000 Hz audio, got {sample_rate}")

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
inputs = processor.feature_extractor(
    audio, sampling_rate=sample_rate, return_tensors="pt"
)
labels = processor.tokenizer(
    str(row["verbatim"]), return_tensors="pt"
).input_ids

print("Audio shape:", audio.shape)
print("Sample rate:", sample_rate)
print("Feature shape:", inputs.input_features.shape)
print("Label shape:", labels.shape)
print("Reference transcript:", row["verbatim"])
print("Preprocessing test passed.")
