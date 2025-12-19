import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer, SFTConfig

# =====================================================
# CONFIG
# =====================================================
model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
dataset_name = "unique_diverse_js_security_dataset.json"
new_model_name = "Qwen2.5-1.5B-Security-Tuned"

torch.backends.cuda.matmul.allow_tf32 = True

# =====================================================
# 1. LOAD BASE MODEL (4-bit QLoRA)
# =====================================================
print(">>> 1. Loading Base Model")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    max_memory={0: "3.5GB", "cpu": "12GB"},
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

model.config.use_cache = False
model.config.pretraining_tp = 1
model.config.torch_dtype = torch.float16

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("✅ Base Model Loaded")

# =====================================================
# 2. APPLY LoRA + SANITIZE TYPES
# =====================================================
print("\n>>> 2. Applying LoRA and Sanitizing Types")

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(model, lora_config)

# Ensure NO BF16 tensors exist
bf16_fixed = 0
for _, param in model.named_parameters():
    if param.dtype == torch.bfloat16:
        param.data = param.data.to(torch.float16)
        bf16_fixed += 1

if bf16_fixed > 0:
    print(f"⚠️ Converted {bf16_fixed} BF16 tensors to FP16")
else:
    print("✅ No BF16 parameters found")

model.print_trainable_parameters()
print("✅ LoRA Applied")

# =====================================================
# 3. LOAD DATASET
# =====================================================
print("\n>>> 3. Loading Dataset")

dataset = load_dataset("json", data_files=dataset_name, split="train")
dataset = dataset.train_test_split(test_size=0.1, seed=42)

train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print("✅ Dataset Loaded")

# =====================================================
# 4. TRAINING CONFIG (TRL CORRECT)
# =====================================================
print("\n>>> 4. Configuring Training Arguments")

training_args = SFTConfig(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,

    optim="paged_adamw_8bit",

    # 🔑 TRL uses eval_strategy (NOT evaluation_strategy)
    eval_strategy="steps",
    eval_steps=50,
    save_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.01,

    # 🔥 AMP COMPLETELY DISABLED (CRASH FIX)
    fp16=False,
    bf16=False,

    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    group_by_length=True,

    report_to="tensorboard",
    dataset_text_field="text",
    packing=False,
)

print("✅ Arguments Configured")

# =====================================================
# 5. INITIALIZE TRAINER
# =====================================================
print("\n>>> 5. Initializing Trainer")

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    args=training_args,
)

print("✅ Trainer Initialized")

# =====================================================
# 6. TRAIN
# =====================================================
print("\n>>> 6. Starting Fine-Tuning")
trainer.train()
print("✅ Training Completed")

# =====================================================
# 7. SAVE ADAPTER
# =====================================================
print("\n>>> 7. Saving Adapter")

trainer.model.save_pretrained(new_model_name)
tokenizer.save_pretrained(new_model_name)

print(f"✅ Adapter saved to '{new_model_name}'")

# =====================================================
# 8. OPTIONAL MERGE (CPU SAFE)
# =====================================================
print("\n>>> 8. Optional Merge")

try:
    del trainer
    torch.cuda.empty_cache()

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    from peft import PeftModel

    merged_model = PeftModel.from_pretrained(base_model, new_model_name)
    merged_model = merged_model.merge_and_unload()

    out_dir = f"{new_model_name}-merged"
    merged_model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    print(f"✅ Merged model saved to '{out_dir}'")

except Exception as e:
    print(f"⚠️ Merge skipped: {e}")

print("\n🎉 ALL DONE — TRAINING IS STABLE")
