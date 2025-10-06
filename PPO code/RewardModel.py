import env_setup
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments
from peft import LoraConfig
from transformers import AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig


# 1. Load the SFT Model and Tokenizer
model = "Qwen/Qwen3-0.6B-Base"

tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForSequenceClassification.from_pretrained(
    model,
    num_labels=1,
    #torch_dtype=torch.float16,
    device_map= None,
    trust_remote_code = True
)



# Pad token must be set - crucial for batch processing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model.config.pad_token_id = tokenizer.pad_token_id

# Verify the pad_token_id is properly set
print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
print(f"Model config pad_token_id: {model.config.pad_token_id}")

# Ensure padding is set to right for sequence classification
tokenizer.padding_side = "right"

# 2. Load and Prepare the Dataset
dataset_name = "Anthropic/hh-rlhf"
train_dataset = load_dataset(dataset_name, data_dir="harmless-base", split="train")
eval_dataset = load_dataset(dataset_name, data_dir="harmless-base", split="test")

def format_dataset(example):
    tokenized_chosen = tokenizer(
        example["chosen"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    tokenized_rejected = tokenizer(
        example["rejected"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    return {
        "input_ids_chosen": tokenized_chosen["input_ids"].squeeze(),
        "attention_mask_chosen": tokenized_chosen["attention_mask"].squeeze(),
        "input_ids_rejected": tokenized_rejected["input_ids"].squeeze(),
        "attention_mask_rejected": tokenized_rejected["attention_mask"].squeeze(),
    }

train_dataset = train_dataset.map(format_dataset)
eval_dataset = eval_dataset.map(format_dataset)

#sample_train = train_dataset.select(range(100))
#sample_eval =  eval_dataset.select(range(10))

# 3. Configure and Set up the Reward Trainer
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="SEQ_CLS",
)

training_args = RewardConfig(
    output_dir="./rm_base_output",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=32,
    #optim="paged_adamw_32bit",
    learning_rate=2e-4,
    eval_strategy="epoch",
    #eval_steps=50,
    save_strategy="epoch",
    #save_steps=50,
    logging_steps=100, #84
    num_train_epochs=2,
    fp16=True,
    bf16=False,
    push_to_hub=False,
    remove_unused_columns=False,
    max_length=512,
    disable_dropout=False,
)

# The trainer correctly receives the base model and the NEW peft_config
trainer = RewardTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,

)

# 4. Start the Training
print("Starting Reward Model Training...")
trainer.train()
print("Reward Model training complete!")

# 5. Save the final Reward Model
trainer.save_model("./reward_base_model")
Print("reward model saved!")