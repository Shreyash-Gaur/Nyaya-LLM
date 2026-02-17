import torch
import json
import random
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datetime import datetime

# ==========================================
# 🎯 SELECT THE MODEL TO EVALUATE
# ==========================================
# Uncomment exactly ONE model you want to test:

ACTIVE_MODEL = "phi4"
# ACTIVE_MODEL = "qwen3_4b"
# ACTIVE_MODEL = "qwen25_7b"
# ACTIVE_MODEL = "gemma3"

CONFIGS = {
    "phi4": {
        "base_model": "microsoft/Phi-4-mini-instruct",
        "adapter_dir": "results/qlora_phase1_phi4_mini"
    },
    "qwen3_4b": {
        "base_model": "Qwen/Qwen3-4B-Instruct-2507",
        "adapter_dir": "results/qlora_phase1_qwen3_4b"
    },
    "qwen25_7b": {
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "adapter_dir": "results/qlora_phase1_qwen25_7b"
    },
    "gemma3": {
        "base_model": "google/gemma-3-4b-it",
        "adapter_dir": "results/qlora_phase1_gemma3_4b"
    }
}

TEST_DATA_PATH = "data/processed/all_acts_test.jsonl"
NUM_SAMPLES = 10  # Increased to 10 for better statistical representation
SEED = 42         # Fixed seed ensures all models face the exact same prompts

def load_test_samples(filepath, num_samples):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    random.seed(SEED)
    return random.sample(data, min(num_samples, len(data)))

def generate_response(model, tokenizer, instruction):
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_output.split("### Response:\n")[-1].strip()

def main():
    cfg = CONFIGS[ACTIVE_MODEL]
    print(f"🚀 Initializing Evaluation for: {ACTIVE_MODEL.upper()}")
    
    # Setup Output File
    os.makedirs("results", exist_ok=True)
    output_filename = f"results/phase1_eval_{ACTIVE_MODEL}.json"
    
    print("Loading 4-bit config...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    print(f"Loading base model: {cfg['base_model']}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True if "qwen" in ACTIVE_MODEL else False,
        dtype=torch.float16
    )

    print(f"Loading LoRA adapter from: {cfg['adapter_dir']}...")
    try:
        model = PeftModel.from_pretrained(base_model, cfg["adapter_dir"])
    except Exception as e:
        print(f"\n❌ ERROR: Could not load adapter. Did {ACTIVE_MODEL} finish training?\n{e}")
        return
        
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"], 
        trust_remote_code=True if "qwen" in ACTIVE_MODEL else False
    )

    print("\n" + "="*60)
    print(f"⚖️ PHASE 1 EVALUATION: {ACTIVE_MODEL.upper()}")
    print("="*60 + "\n")

    samples = load_test_samples(TEST_DATA_PATH, num_samples=NUM_SAMPLES)
    results = []

    for i, sample in enumerate(samples, 1):
        instruction = sample["instruction"]
        ground_truth = sample["output"]
        
        print(f"[{i}/{NUM_SAMPLES}] Generating prediction...")
        prediction = generate_response(model, tokenizer, instruction)
        
        # Save to list
        results.append({
            "model": ACTIVE_MODEL,
            "instruction": instruction,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        })

        # Optional: Print to console for real-time monitoring
        print(f"✅ Saved prediction {i}")

    # Write full results to JSON file
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\n" + "="*60)
    print(f"💾 RESULTS SAVED TO: {output_filename}")
    print("="*60)

if __name__ == "__main__":
    main()