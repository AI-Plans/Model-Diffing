# Model Diffing

Experiments in language model fine-tuning and model diffing, centred on Qwen3-0.6B. This shared repository brings together training notebooks for several alignment methods and a crosscoder notebook for comparing activations before and after fine-tuning.

For the model comparison work, start with [the crosscoder notebook](Crosscoder/cross-coder.ipynb). For training experiments, use the guide below.

## Crosscoder analysis

The [crosscoder notebook](Crosscoder/cross-coder.ipynb) compares `Qwen/Qwen3-0.6B` with the GRPO checkpoint `AIPlans/Qwen3-0.6B-GRPO-RM_NVIDIA`. It includes:

- Activation collection with TransformerLens at `blocks.14.hook_resid_pre`.
- A crosscoder with BatchTopK sparsity and an alternative L1 configuration.
- Activation buffering, checkpoint saving, and Weights & Biases logging.
- Plots of relative decoder norms and cosine similarity between decoder vectors.

The active token-loading path uses Wikitext and LMSYS conversations. A separate helper prepares HelpSteer2 prompts. These are exploratory notebooks; the plots should be read alongside the configuration and saved outputs of each run.

## Notebook guide

| Area | Files | Contents |
| --- | --- | --- |
| Model diffing | [Crosscoder](Crosscoder/cross-coder.ipynb) | Activation comparison between the original Qwen3-0.6B model and a GRPO checkpoint. |
| DPO | [LoRA](DPOTrainer/dpo-trainer-model-diffing.ipynb) · [Full model](DPOTrainer/dpo-trainer-optimized-fullmodel-training.ipynb) | HelpSteer2 preference training; the full-model notebook includes a two-GPU Accelerate/DeepSpeed configuration. |
| KTO | [KTO trainer](KTO-Trainer/kto-trainer.ipynb) | Binary feedback prepared from HelpSteer2 helpfulness scores, with class balancing and LoRA training. |
| ORPO | [ORPO trainer](ORPOTrainer/orpo-training.ipynb) | LoRA training on cleaned HelpSteer2 preference pairs. |
| GRPO | [GRPO trainer](DPRO-Trainer/grpo-trainer-with-wandb.ipynb) | LoRA training with NVIDIA reward-model API scores and Weights & Biases logging. |
| IPO | [IPO trainer](IPO/IPO_trainer.ipynb) | Preference training using the IPO loss in TRL's DPOTrainer. |
| Supervised fine-tuning | [SFT notebooks](sft/) | Qwen3-0.6B experiments using HH-RLHF and HelpSteer2 data. |
| HH-RLHF DPO | [Full-model DPO](dpo-qwen0-6b-fft/dpo_qwen0_6b_fft.ipynb) | A separate DPO experiment on Anthropic HH-RLHF. |
| PPO | [PPO notebooks and scripts](PPO%20code/) | Supervised fine-tuning, reward-model training, and PPO. |
| Reward modelling and ReMax | [Reward model](Reward%20Models/Reward%20Model/Reward_Model_trainer.ipynb) · [ReMax](Reward%20Models/ReMax-Trainer/ReMax_trainer.ipynb) | Reward-model training and a ReMax training loop. |
| RedDebate | [Notebook and data](RedDebate/) | A RedDebate experiment with debate logs, exported embeddings, and a 100-question HarmBench subset. |

The GRPO notebook is stored in `DPRO-Trainer/`, retaining the existing folder name.

## Running an experiment

The notebooks contain their own installation and configuration cells. Several use Kaggle or Colab paths, saved checkpoints, or account-specific settings, so review those cells before running them in a new environment.

1. Open the notebook for the experiment you want to run in a GPU-enabled environment.
2. Review its dependency installation cells. There is no shared, pinned environment for the repository.
3. Set the model and dataset IDs, local paths, and output locations for your run. For a fresh run, remove or update any `resume_from_checkpoint` argument pointing to a previous run.
4. Configure the services used by that notebook: Hugging Face, Weights & Biases, and the NVIDIA reward API where applicable. Keep credentials in your environment or notebook platform's secret storage.
5. Review the model upload cells before executing them; several notebooks point to the `AIPlans` Hugging Face organisation.

Core libraries used across the experiments include PyTorch, Transformers, TRL, Datasets, PEFT, and bitsandbytes. The crosscoder also uses TransformerLens; the two-GPU DPO configuration uses Accelerate and DeepSpeed.

## Training notes

These notes record issues encountered during the experiments. They are not a compatibility matrix for current library versions.

### KTO

Earlier runs encountered compatibility issues with PyTorch 2.8.0 and later, as well as high memory use. The original workaround was to use an earlier Colab runtime and keep the PyTorch-related packages compatible. Kaggle did not require the same runtime change in those runs.

References retained from the experiments: [Hugging Face Discord discussion](https://discord.com/channels/879548962464493619/879548962464493622/1440675714419523585) and [KTO memory notes](https://huggingface.co/datasets/John6666/forum2/blob/main/trl_kto_blow_up_memory_1.md).

### ORPO

An earlier run encountered a `zip(..., strict=...)` error. Updating the libraries resolved it in that environment.

### GRPO

Use the logged rewards, gradient norms, and completion statistics alongside the reported loss when inspecting a run. A displayed loss of zero alone is not enough to determine whether training is working.

See the [GRPO run notes](DPRO-Trainer/README.md) for the memory settings used here. Earlier troubleshooting references: [zero-loss discussion 1](https://huggingface.co/datasets/John6666/forum3/blob/main/grpo_0_loss_issue_1.md) and [zero-loss discussion 2](https://huggingface.co/datasets/John6666/forum3/blob/main/grpo_0_loss_issue_2.md).

## Project scope and contributions

This repository collects experiments from multiple contributors. Individual notebooks contain their own configurations and outputs; there is no consolidated benchmark comparing all the methods under a single evaluation setup.

Contribution history is available in the [repository commits](https://github.com/AI-Plans/Model-Diffing/commits/main/). The experiments use existing models, datasets, libraries, and methods. The RedDebate notebook builds on the [RedDebate project](https://github.com/aliasad059/RedDebate), and the crosscoder includes a normalization routine marked as adapted from SAELens.
