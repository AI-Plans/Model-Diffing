import env_setup
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

# 1. Load the Pre-trained BASE Model and Tokenizer

model_name = "Qwen/Qwen3-0.6B-Base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map= None,
    trust_remote_code = True
)
#uncomment this when using full dataset
model.gradient_checkpointing_enable()

# Base models often don't have a pad token, so we set it here.
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id


# 2. Load and Prepare the Dataset
dataset_name = "Anthropic/hh-rlhf"
dataset = load_dataset(dataset_name, data_dir="harmless-base", split="train")
#test_dataset = load_dataset(dataset_name, data_dir='harmless-base', split='test')

def format_dataset_for_base_model(example):
    return {"text": example["chosen"]}


dataset = dataset.map(format_dataset_for_base_model)



# 4.The SFT Trainer
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = TrainingArguments(
    output_dir="./sft_base_output",
    per_device_train_batch_size=16, 
    gradient_accumulation_steps=32,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    logging_steps=100,  
    num_train_epochs=2,
    #max_steps=100,
    fp16=True,
    bf16= False,
    push_to_hub=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset= dataset,
    peft_config=peft_config,
    args=training_args,
)

# 5. Start the Training
print("Starting Supervised Fine-Tuning on the BASE model...")
trainer.train()
print("SFT on base model complete!")

# Save the new SFT model
trainer.save_model("./sft_base_model")
print("Model Saved!")