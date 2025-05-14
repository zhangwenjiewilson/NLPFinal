# scripts/download_streaming.py
from datasets import load_dataset
ds = load_dataset(
        "pierreguillou/DocLayNet-small",      # or DocLayNet-base
        split="train",
        streaming=True,
        trust_remote_code=True)
# iterating once forces the lazy download into ~/.cache/huggingface/
for _ in ds.take(1):
    print("✓ dataset cached – ready for stitching")
