import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATA_PATH = "data/processed/all_acts_train.jsonl"
VAL_PATH = "data/processed/all_acts_val.jsonl"
OUTPUT_DIR = "results/lora_phase1_qwen25_7b"
MAX_SEQ_LENGTH = 512


def format_prompt(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"


def main():
    print("Loading dataset...")
    dataset = load_dataset(
        "json", data_files={"train": DATA_PATH, "validation": VAL_PATH}
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model in fp16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,    # replaces deprecated torch_dtype
        device_map="auto",      # replaces manual .to("cuda"), handles multi-GPU/CPU offload too
        trust_remote_code=True,
    )

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    print("Applying LoRA adapters...")
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("Setting up Trainer...")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        eval_strategy="steps",
        eval_steps=200,
        logging_steps=50,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        optim="adamw_torch",
        save_strategy="epoch",
        save_total_limit=2,
        max_length=MAX_SEQ_LENGTH,
        report_to="none",
        dataloader_pin_memory=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        formatting_func=format_prompt,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model adapter...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Qwen2.5-7B LoRA Training Complete.")


if __name__ == "__main__":
    main()