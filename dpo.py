import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from trl import DPOTrainer, DPOConfig

# Load training data
with open("training_f.json") as f:
    raw = json.load(f)

data = {
    "prompt":   [item["prompt"]   for item in raw],
    "chosen":   [item["chosen"]   for item in raw],
    "rejected": [item["rejected"] for item in raw]
}
dataset = Dataset.from_dict(data)
print(f"Training on {len(dataset)} pairs")

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load model and tokenizer
MODEL = "unsloth/Llama-3.2-3B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading model in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb_config,
    device_map={'': torch.cuda.current_device()}
)

model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

# LoRA config
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Reference model — frozen copy of the base model
print("Loading reference model...")
ref_model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb_config,
    device_map={'': torch.cuda.current_device()}
)

# DPO Training
training_args = DPOConfig(
    output_dir="./dpo_model",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    learning_rate=5e-5,
    beta=0.1,
    max_length=256,
    max_prompt_length=128,
    fp16=True,
    logging_steps=20,
    save_strategy="no",
    report_to="none",
    optim="paged_adamw_8bit"
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

print("Starting DPO training...")
trainer.train()
print("Training complete.")

model.save_pretrained("./dpo_model")
tokenizer.save_pretrained("./dpo_model")
print("Model saved to ./dpo_model")

# Inference on test set
print("\nRunning inference on test set...")

with open("test_f.json") as f:
    test_data = json.load(f)

model.eval()
results = []

for item in test_data:
    print(f"  Prompt {item['id']}...")

    inputs = tokenizer(
        item["prompt"] + " Answer in 3 sentences maximum.",
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    results.append({
        "id":       item["id"],
        "type":     item["type"],
        "prompt":   item["prompt"],
        "response": response.strip()
    })

with open("dpo_responses.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Done. Saved to dpo_responses.json")