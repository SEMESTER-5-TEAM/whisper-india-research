
import torch
from transformers import WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model

MODEL_NAME = "openai/whisper-small"

print("Loading Whisper-small onto CPU...")
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())

print(f"Total parameters: {total:,}")
print(f"Trainable parameters: {trainable:,}")
print(f"Trainable percentage: {100 * trainable / total:.4f}%")
print("LoRA setup test passed.")
