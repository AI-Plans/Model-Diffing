# GRPO training notes

The [GRPO notebook](grpo-trainer-with-wandb.ipynb) fine-tunes Qwen3-0.6B with LoRA on HelpSteer2 prompts. It obtains rewards through NVIDIA's API using `nvidia/llama-3.1-nemotron-70b-reward` and logs training metrics to Weights & Biases.

## Before running

- Configure Hugging Face, Weights & Biases, and NVIDIA API credentials using your environment or notebook platform's secret storage.
- Update `resume_from_checkpoint` to an existing checkpoint, or remove it for a fresh run.
- Review the output paths and Hugging Face destination before running the save and upload cells.

## Memory settings used in the notebook

| Setting | Current configuration |
| --- | --- |
| Prompt filtering | Keeps prompts at or below the 90th percentile of token length. |
| Batch size | `per_device_train_batch_size=4` |
| Gradient accumulation | `gradient_accumulation_steps=6` |
| Completions per prompt | `num_generations=4` |
| Completion length | `max_completion_length=248` |
| Qwen thinking mode | `chat_template_kwargs={"enable_thinking": False}` |

Adjust these settings to fit the available GPU memory. Use the Weights & Biases logs to inspect reward behaviour, gradients, and completion truncation when changing the configuration. A displayed loss of zero on its own does not establish whether a run is training correctly.

[Back to the project overview](../README.md)
