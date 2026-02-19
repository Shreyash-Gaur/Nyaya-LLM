import torch
import json
import requests
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==============================
# CONFIG
# ==============================
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507" # The base model you used on Kaggle
PHASE_1_ADAPTER = "path/to/downloaded/phase1_qlora_adapter"
PHASE_2_ADAPTER = "path/to/downloaded/phase2_qlora_adapter"
TEST_FILE = "data/eval/eval_set.json"
OUTPUT_FILE = "data/eval/evaluation_results.json"

# Ollama Judge Config
JUDGE_MODEL = "qwen2.5:7b" # Or llama3 if you have it
OLLAMA_URL = "http://localhost:11434/api/generate"

# ==============================
# THE JUDGE PROMPT
# ==============================
JUDGE_SYSTEM_PROMPT = """
You are an impartial, strict legal AI judge evaluating another AI's output.
Compare the Model Answer to the Ground Truth Reference.

Score the answer from 1 to 5 based on:
1: Completely wrong, hallucinated, or legally dangerous.
2: Poor, misses key legal concepts.
3: Acceptable, gets the core idea right but lacks precision.
4: Good, accurate legal reasoning and application.
5: Perfect, flawless legal interpretation and application.

Respond ONLY with a valid JSON object in this exact format:
{
  "score": <int>,
  "reasoning": "<short explanation>"
}
"""

def generate_answer(model, tokenizer, prompt):
    inputs = tokenizer(f"### Instruction:\n{prompt}\n\n### Response:\n", return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
    
    # Strip out the prompt text to get just the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:\n")[-1].strip()

def score_with_judge(prompt, reference, model_answer):
    judge_prompt = f"""
    USER PROMPT: {prompt}
    GROUND TRUTH REFERENCE: {reference}
    MODEL ANSWER: {model_answer}
    
    Evaluate the MODEL ANSWER against the GROUND TRUTH. Output JSON only.
    """
    
    payload = {
        "model": JUDGE_MODEL,
        "prompt": JUDGE_SYSTEM_PROMPT + judge_prompt,
        "temperature": 0.0,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload).json()["response"]
        match = re.search(r"\{.*\}", response, re.DOTALL)
        result = json.loads(match.group())
        return result["score"], result["reasoning"]
    except Exception as e:
        print(f"Judge failed: {e}")
        return 0, "Judge error"

def main():
    with open(TEST_FILE, "r") as f:
        test_data = json.load(f)

    print(f"Loading Base Model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
    )

    results = []

    for phase_name, adapter_path in [("Phase_1", PHASE_1_ADAPTER), ("Phase_2", PHASE_2_ADAPTER)]:
        print(f"\n--- Loading {phase_name} Adapter ---")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()

        print(f"Generating and Scoring answers for {phase_name}...")
        for item in tqdm(test_data):
            # 1. Model Generates Answer
            answer = generate_answer(model, tokenizer, item["prompt"])
            
            # 2. Judge Scores Answer
            score, reasoning = score_with_judge(item["prompt"], item["reference"], answer)
            
            # 3. Save Result
            results.append({
                "model": phase_name,
                "category": item["category"],
                "prompt": item["prompt"],
                "answer": answer,
                "score": score,
                "judge_reasoning": reasoning
            })
            
        # Unload adapter to make room for the next one
        model.unload()

    # Save final results
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    # Print Summary Aggregation
    print("\n================ EVALUATION SUMMARY ================")
    for phase in ["Phase_1", "Phase_2"]:
        print(f"\n{phase} Scores:")
        scores = [r for r in results if r["model"] == phase]
        
        # Calculate overall average
        overall = sum(r["score"] for r in scores) / len(scores)
        print(f"  Overall Average: {overall:.2f} / 5.0")
        
        # Calculate average by category
        categories = set(r["category"] for r in scores)
        for cat in categories:
            cat_scores = [r["score"] for r in scores if r["category"] == cat]
            cat_avg = sum(cat_scores) / len(cat_scores)
            print(f"    - {cat}: {cat_avg:.2f} / 5.0")

if __name__ == "__main__":
    main()