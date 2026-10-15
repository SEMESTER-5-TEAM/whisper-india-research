import pandas as pd
import io
import soundfile as sf
import whisper
import torch

# Load first Svarah record
df = pd.read_parquet(
    "data/raw/svarah/data/test-00000-of-00003.parquet"
)

x = df.iloc[0]["audio_filepath"]

# Decode embedded audio
audio, sr = sf.read(
    io.BytesIO(x["bytes"]),
    dtype="float32"
)

print("Sample rate:", sr)
print("Audio duration:", len(audio) / sr)
print("Reference:", df.iloc[0]["text"])

# Load Whisper
model = whisper.load_model("small", device="cuda")

# Prepare audio exactly as Whisper expects
audio = whisper.pad_or_trim(audio)

mel = whisper.log_mel_spectrogram(
    audio,
    n_mels=model.dims.n_mels
).to(model.device)

# Detect language
_, probs = model.detect_language(mel)

print("\nTop detected languages:")

for lang, probability in sorted(
    probs.items(),
    key=lambda x: x[1],
    reverse=True
)[:10]:
    print(f"{lang}: {probability:.6f}")

# Decode using the detected language
detected_language = max(probs, key=probs.get)

print("\nSelected language:", detected_language)

options = whisper.DecodingOptions(
    language=detected_language,
    temperature=0
)

result = whisper.decode(model, mel, options)

print("Decoded text:", result.text)