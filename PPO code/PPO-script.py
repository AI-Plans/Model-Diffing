import env_setup
import warnings, torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers.modeling_outputs import CausalLMOutputWithPast
import pandas as pd
import os

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore")

# --- 1. A Fully Patched Wrapper Class ---
class PatchedWithValueHead(AutoModelForCausalLMWithValueHead):
    def __init__(self, base_model, **kwargs):
        super().__init__(base_model, **kwargs)
        self.config = base_model.config
        self.generation_config = base_model.generation_config
        self.base_model_prefix = getattr(base_model, "base_model_prefix", "")
        if self.base_model_prefix:
            backbone = getattr(base_model, self.base_model_prefix)
            setattr(self, self.base_model_prefix, backbone)
        self.prepare_inputs_for_generation = base_model.prepare_inputs_for_generation

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ):
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        value = self.v_head(outputs.hidden_states[-1]).squeeze(-1)
        outputs.value = value
        return outputs

    def score(self, hidden_states):
        return self.v_head(hidden_states).squeeze(-1)

    @property
    def is_gradient_checkpointing(self) -> bool:
        return (
            getattr(self.pretrained_model, "is_gradient_checkpointing", False)
            or getattr(self.pretrained_model, "gradient_checkpointing", False)
            or getattr(self.pretrained_model, "_gradient_checkpointing", False)
        )

    @property
    def is_peft_model(self):
        return isinstance(self.pretrained_model, PeftModel)

# --- 2. Configuration ---
config = PPOConfig(
    learning_rate=1.41e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=32,
    num_ppo_epochs=2,
    max_grad_norm=1.0,
    seed=42,
    kl_coef=0.05,
    cliprange=0.2,
    num_mini_batches=4,
    #total_episodes=100,
    response_length=100,
    temperature=0.7,
    num_train_epochs=2,  
    eval_strategy="epoch",
    sft_model_path="./sft_base_model",
    reward_model_path="./reward_base_model",
)

# --- 3. Load Tokenizer ---
model_name = config.sft_model_path
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# --- 4. Load Models using the Fully Patched Class ---
print("Loading models...")

# A. Load the base SFT model and apply LoRA adapters if they exist
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
try:
    base_model = PeftModel.from_pretrained(base_model, model_name)
    print("LoRA adapters loaded successfully.")
except Exception as e:
    print(f"No LoRA adapters found: {e}")

# B. Create the PPO models
print("Creating policy, value, and reference models...")
policy_model = PatchedWithValueHead(base_model)
value_model = PatchedWithValueHead(base_model)
ref_model = PatchedWithValueHead(base_model)
print("Models created successfully.")

# C. Load the reward model
print("Loading reward model...")
reward_model = AutoModelForSequenceClassification.from_pretrained(
    config.reward_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    num_labels=1,
)
print("Reward model loaded successfully.")

# --- 5. Prepare Dataset and Data Collator ---
def prepare_and_tokenize_dataset(example):
    conversation = example["chosen"]
    prompt = conversation.split("\n\nAssistant:")[0] + "\n\nAssistant:"
    prompt = prompt.strip()[:400]
    tokenized_prompt = tokenizer(prompt, truncation=True, max_length=config.response_length, add_special_tokens=False)
    tokenized_prompt["query"] = prompt
    return tokenized_prompt

class PPODataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, data):
        queries = [d.pop("query") for d in data]
        batch = self.tokenizer.pad(data, padding=True, return_tensors="pt")
        batch["query"] = queries
        return batch

print("Preparing dataset...")
tr_dataset = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="train")
te_dataset = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="test")

train_dataset = tr_dataset.map(prepare_and_tokenize_dataset)
test_dataset = te_dataset.map(prepare_and_tokenize_dataset)

train_dataset = train_dataset.remove_columns(tr_dataset.column_names)
test_dataset = test_dataset.remove_columns(te_dataset.column_names)


train_dataset_ppo = train_dataset 
eval_dataset_ppo = test_dataset


data_collator = PPODataCollator(tokenizer=tokenizer)
print("Dataset and collator are ready.")

# --- 6. Initialize PPOTrainer and Start Training ---
print("Initializing PPOTrainer...")
try:
    ppo_trainer = PPOTrainer(
        args=config,
        processing_class=tokenizer,
        model=policy_model,
        ref_model=ref_model,
        value_model=value_model,
        reward_model=reward_model,
        train_dataset=train_dataset_ppo,
        eval_dataset=eval_dataset_ppo, # Keep as empty list if not evaluating during PPO loop
        data_collator=data_collator,
    )
    print("PPOTrainer initialized successfully!")

    print("Starting PPO training...")
    ppo_trainer.train()
    print("PPO Training complete!")

    print("Saving final PPO model...")
    os.makedirs("./ppo_final_model", exist_ok=True)
    policy_model.save_pretrained("./ppo_final_model", use_safetensors=False)
    tokenizer.save_pretrained("./ppo_final_model")
    print("Model and tokenizer saved successfully!")

    # --- 7. Custom Evaluation ---
    # print("Performing custom evaluation...")
    # conversations = []
    # scores = []
    # device = next(policy_model.parameters()).device  # Get device from model

    # for example in eval_dataset_ppo:
    #     # Convert to tensors
    #     input_ids = torch.tensor(example["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
    #     attention_mask = torch.tensor(example["attention_mask"], dtype=torch.long).unsqueeze(0).to(device)

    #     # Generate response
    #     with torch.no_grad():
    #         generated_ids = policy_model.generate(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             num_return_sequences=1,
    #             pad_token_id=tokenizer.pad_token_id,
    #             eos_token_id=tokenizer.eos_token_id,
    #         )

    #     # Get response text
    #     response_ids = generated_ids[:, len(input_ids[0]):]
    #     response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    #     # Get query text
    #     query_text = example["query"].split("\n\nAssistant:")[0]

    #     # Combine into conversation
    #     conversation_text = f"{query_text}\nAssistant: {response_text}"
    #     conversations.append(conversation_text)

    #     # Compute score
    #     full_text = f"{example['query']}{response_text}"
    #     reward_inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512, add_special_tokens=False).to(device)
    #     with torch.no_grad():
    #         reward_output = reward_model(**reward_inputs)
    #     score = reward_output.logits[0, 0].item()
    #     scores.append(score)

    # # Create DataFrame
    # df = pd.DataFrame({
    #     "conversation": conversations,
    #     "score": scores
    # })



except Exception as e:
    print(f"\n--- ERROR DURING PPO TRAINING ---")
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Message: {e}")
    import traceback
    traceback.print_exc()