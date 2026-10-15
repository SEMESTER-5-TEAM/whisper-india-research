
from collections import defaultdict
from huggingface_hub import HfApi, get_token

REPO_ID = "ai4bharat/IndicVoices"

# Check language configurations relevant to our target regions.
CONFIGS = [
    "assamese",
    "bengali",
    "kashmiri",
    "urdu",
    "tamil",
]

api = HfApi(token=get_token())

print("Retrieving IndicVoices repository file metadata...")
print("No audio files will be downloaded.\n")

files_by_config = defaultdict(list)

for item in api.list_repo_tree(
    repo_id=REPO_ID,
    repo_type="dataset",
    recursive=True,
    expand=True,
):
    path = item.path
    parts = path.split("/")

    if len(parts) < 2:
        continue

    config = parts[0]

    if config in CONFIGS and path.endswith(".parquet"):
        size = getattr(item, "size", None)
        files_by_config[config].append((path, size))

for config in CONFIGS:
    files = files_by_config[config]
    total_bytes = sum(size or 0 for _, size in files)

    print(f"\n{'=' * 55}")
    print(f"CONFIGURATION: {config}")
    print(f"Parquet files found: {len(files)}")
    print(f"Total listed size: {total_bytes / (1024**3):.2f} GiB")

    for path, size in files:
        size_gib = size / (1024**3) if size is not None else None
        size_text = f"{size_gib:.3f} GiB" if size_gib is not None else "size unknown"
        print(f"  {path} — {size_text}")

print("\nInspection complete. Only repository file metadata was requested.")
