import json
import random

original_file = "data/processed/all_acts_train.jsonl"
augmented_file = "data/agumented/all_acts_train_augmented.jsonl"
output_file = "data/mixed/nyaya_phase2_mixed_train.jsonl"

mixed_data = []

print("Loading datasets...")

# 1. Load Original Data (The raw statutes)
with open(original_file, 'r', encoding='utf-8') as f:
    mixed_data.extend([json.loads(line) for line in f])

# 2. Load Augmented Data (The new synthetic Q&A)
with open(augmented_file, 'r', encoding='utf-8') as f:
    mixed_data.extend([json.loads(line) for line in f])

# 3. Shuffle the combined dataset
print("Shuffling data to prevent catastrophic forgetting...")
random.seed(42)  
random.shuffle(mixed_data)

# 4. Save to new JSONL
print("Saving mixed dataset...")
with open(output_file, 'w', encoding='utf-8') as f:
    for item in mixed_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"\n✅ Successfully mixed and shuffled!")
print(f"📊 Total rows ready for Phase 2: {len(mixed_data)}")
print(f"💾 Saved to: {output_file}")