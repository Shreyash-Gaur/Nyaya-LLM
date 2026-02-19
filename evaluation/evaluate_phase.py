import torch
import json
import re
import os
import requests
import random
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datetime import datetime

# ==========================================
# 🎯 SELECT THE MODEL TO EVALUATE
# ==========================================
ACTIVE_MODEL = "phi4_qlora"
# ACTIVE_MODEL = "phi4_lora"
# ACTIVE_MODEL = "qwen3_qlora"
# ACTIVE_MODEL = "qwen3_lora"
# ACTIVE_MODEL = "gemma3_qlora"
# ACTIVE_MODEL = "gemma3_lora"

CONFIGS = {
    "phi4_qlora":   {"base_model": "microsoft/Phi-4-mini-instruct", "adapter_dir": "results/qlora_phase1_phi4_mini"},
    "phi4_lora":    {"base_model": "microsoft/Phi-4-mini-instruct", "adapter_dir": "results/lora_phase1_phi4_mini"},
    "qwen3_qlora":  {"base_model": "Qwen/Qwen3-4B-Instruct-2507",  "adapter_dir": "results/qlora_phase1_qwen3_4b"},
    "qwen3_lora":   {"base_model": "Qwen/Qwen3-4B-Instruct-2507",  "adapter_dir": "results/lora_phase1_qwen3_4b"},
    "gemma3_qlora": {"base_model": "google/gemma-3-4b-it",          "adapter_dir": "results/qlora_phase1_gemma3_4b"},
    "gemma3_lora":  {"base_model": "google/gemma-3-4b-it",          "adapter_dir": "results/lora_phase1_gemma3_4b"},
}

# ==========================================
# ⚙️  CONFIG
# ==========================================
TEST_DATA_PATH = "data/processed/all_acts_test.jsonl"
NUM_SAMPLES    = 50     # Fixed seed guarantees all models see the same 50
SEED           = 42
RESULTS_DIR    = "results"
OLLAMA_URL     = "http://localhost:11434/api/generate"
JUDGE_MODEL    = "qwen2.5:7b"
JUDGE_RETRIES  = 3


# ==========================================
# 🏷️  TASK TYPE DETECTOR
# Buckets each sample by instruction style
# so the summary breakdown is meaningful.
# ==========================================
def detect_task_type(instruction: str) -> str:
    il = instruction.lower()
    if il.startswith("explain"):
        return "Explanation"
    elif il.startswith("summarize"):
        return "Summarization"
    elif il.startswith("what does"):
        return "Direct Q&A"
    elif il.startswith("under which act"):
        return "Act Identification"
    else:
        return "Other"


# ==========================================
# 📦  LOAD TEST SAMPLES
# Same 50 questions for every model run.
# ==========================================
def load_test_samples(filepath: str, num_samples: int) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    random.seed(SEED)
    return random.sample(data, min(num_samples, len(data)))


# ==========================================
# 🤖  GENERATION
# ==========================================
def generate_response(model, tokenizer, instruction: str) -> str:
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


# ==========================================
# 🧑‍⚖️  JUDGE SCORING
# Sends each (instruction, reference,
# prediction) triple to local Ollama and
# extracts a 1–5 score with reasoning.
# ==========================================
JUDGE_PROMPT_TEMPLATE = """You are a strict legal AI evaluator. Score the MODEL ANSWER against the REFERENCE ANSWER.

SCORING RUBRIC:
  5 - Perfect. Legally accurate, complete, no errors.
  4 - Good. Correct core content, minor omissions or slight imprecision.
  3 - Acceptable. Gets the general idea but misses important legal details.
  2 - Poor. Partially correct but contains notable legal errors.
  1 - Wrong. Fabricated law, completely incorrect, or irrelevant answer.

IMPORTANT:
- Score based on semantic correctness, NOT exact wording match.
- The reference may be long. The model only needs to capture the key legal meaning.
- If the model answer is legally equivalent to the reference but phrased differently, score it high.

QUESTION:
{instruction}

REFERENCE ANSWER:
{reference}

MODEL ANSWER:
{prediction}

Respond ONLY with a valid JSON object, nothing else:
{{"score": <int 1-5>, "reasoning": "<one concise sentence>"}}"""


def judge_score(instruction: str, reference: str, prediction: str) -> tuple:
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        instruction=instruction,
        reference=reference[:600],    # Trim long statute text to avoid token overflow
        prediction=prediction[:600]
    )

    payload = {
        "model": JUDGE_MODEL,
        "prompt": prompt,
        "temperature": 0.0,
        "stream": False
    }

    for attempt in range(JUDGE_RETRIES):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=60)
            response.raise_for_status()
            raw = response.json()["response"]

            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in judge response")

            parsed = json.loads(match.group())
            score = int(parsed["score"])
            reasoning = parsed.get("reasoning", "")

            if not (1 <= score <= 5):
                raise ValueError(f"Score {score} is out of valid range 1–5")

            return score, reasoning

        except Exception as e:
            print(f"      ⚠️  Judge attempt {attempt + 1} failed: {e}")
            if attempt == JUDGE_RETRIES - 1:
                return 0, "Judge error — skipped"

    return 0, "Judge error — skipped"


# ==========================================
# 📊  SUMMARY PRINTER
# ==========================================
def print_summary(results: list, model_name: str):
    valid = [r for r in results if r["score"] > 0]

    if not valid:
        print("No valid scores to summarise.")
        return

    overall_avg = sum(r["score"] for r in valid) / len(valid)

    by_task = defaultdict(list)
    for r in valid:
        by_task[r["task_type"]].append(r["score"])

    score_dist = defaultdict(int)
    for r in valid:
        score_dist[r["score"]] += 1

    print("\n" + "=" * 60)
    print(f"📊  PHASE 1 SUMMARY — {model_name.upper()}")
    print("=" * 60)
    print(f"  Scored      : {len(valid)} / {len(results)} samples")
    print(f"  Overall avg : {overall_avg:.2f} / 5.0")

    print("\n  Score distribution:")
    for s in range(5, 0, -1):
        count = score_dist.get(s, 0)
        bar = "█" * count
        print(f"    {s}/5  {bar} ({count})")

    print("\n  By task type:")
    for task in sorted(by_task.keys()):
        scores = by_task[task]
        avg = sum(scores) / len(scores)
        print(f"    {task:<20} avg={avg:.2f}  (n={len(scores)})")

    print("=" * 60)


# ==========================================
# 🚀  MAIN
# ==========================================
def main():
    cfg = CONFIGS[ACTIVE_MODEL]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_file = f"{RESULTS_DIR}/phase1_eval_{ACTIVE_MODEL}.json"

    # ── Load model ─────────────────────────────
    print(f"\n🚀 Evaluating: {ACTIVE_MODEL.upper()}")
    print("Loading 4-bit quantization config...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    print(f"Loading base model: {cfg['base_model']}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=("qwen" in ACTIVE_MODEL),
        torch_dtype=torch.float16
    )

    print(f"Loading LoRA adapter: {cfg['adapter_dir']}...")
    try:
        model = PeftModel.from_pretrained(base_model, cfg["adapter_dir"])
        model.eval()
    except Exception as e:
        print(f"\n❌ Could not load adapter. Did training finish?\n{e}")
        return

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"],
        trust_remote_code=("qwen" in ACTIVE_MODEL)
    )

    # ── Load eval data ──────────────────────────
    print(f"\nLoading {NUM_SAMPLES} test samples (seed={SEED})...")
    samples = load_test_samples(TEST_DATA_PATH, NUM_SAMPLES)

    print(f"\n{'='*60}")
    print(f"⚖️   PHASE 1 EVALUATION: {ACTIVE_MODEL.upper()}")
    print(f"{'='*60}\n")

    results = []

    for i, sample in enumerate(samples, 1):
        instruction  = sample["instruction"]
        ground_truth = sample["output"]
        task_type    = detect_task_type(instruction)

        print(f"[{i:02d}/{NUM_SAMPLES}] [{task_type}]")
        print(f"  Prompt : {instruction[:80]}...")

        # Step 1 — Generate answer
        prediction = generate_response(model, tokenizer, instruction)
        print(f"  Answer : {prediction[:80]}...")

        # Step 2 — Judge scores it
        score, reasoning = judge_score(instruction, ground_truth, prediction)
        print(f"  Score  : {score}/5  — {reasoning}\n")

        results.append({
            "model":        ACTIVE_MODEL,
            "task_type":    task_type,
            "instruction":  instruction,
            "ground_truth": ground_truth,
            "prediction":   prediction,
            "score":        score,
            "reasoning":    reasoning,
            "timestamp":    datetime.now().isoformat()
        })

    # ── Save raw results ────────────────────────
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"💾 Raw results saved → {output_file}")

    # ── Print summary ───────────────────────────
    print_summary(results, ACTIVE_MODEL)


if __name__ == "__main__":
    main()