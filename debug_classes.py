# debug_classes.py
# Run this to verify class mapping and split sizes before training.
from collections import Counter

from config import CLASS_INDICES, PREDICTION_LABELS
from cli.data import split_sizes, load_split

print("=" * 60)
print("🔍 CLASS MAPPING DEBUG")
print("=" * 60)

sizes = split_sizes()
print(f"\nClass Indices (config): {CLASS_INDICES}")
print(f"Prediction labels     : {PREDICTION_LABELS}")
print("\nSeeded split sizes (files per class):")
for split, counts in sizes.items():
    total = sum(counts.values())
    print(f"  {split:<5}: {total:>5} files  {counts}")

print("\nGenerator sanity check (train split, first batch):")
gen = load_split("train", batch_size=32, augment=True, seed=42)
print(f"  Class indices: {gen.class_indices}")
print(f"  Train samples: {len(gen.filenames)}")
dist = Counter(gen.classes)
for name, idx in gen.class_indices.items():
    print(f"  {name} (class {idx}): {dist[idx]} images")

print("\n" + "=" * 60)
print("✅ Verify class_indices matches CLASS_INDICES in config.py")
print("=" * 60)
