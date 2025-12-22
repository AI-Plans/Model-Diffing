Tips 
1) Connect to wandb to log metrics. This can help for debugging.
To manage GPU memory
1) Clean dataset to remove huge prompts
2) Tune per_device_train_batch_size and gradient_accumulation_steps
3) GRPO generates answers many times for every prompt. This can be controlled to reduce space. num_generations=4
4) Disable thinking. For Qwen3 model training pass this chat_template_kwargs={"enable_thinking": False} to the config
5) Control max_completion_length. This can be tuned using wandb to see the truncation percentage. 
