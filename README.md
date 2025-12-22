## KTO Trainer
Things to be mindful of for KTOTraining 
1) KTOTrainer does not work for latest versions of torch( 2.8.0 onwards). When downgrading torch, we need to downgrade torchvision and torchtext as well. Thankfully, You can change the runtime version in GPU selection section of colab. Choose the penultimate version. If you are working in Kaggle, no changes are needed.

2) KTOTrainer takes a lot of space . Refer-https://discord.com/channels/879548962464493619/879548962464493622/1440675714419523585 (Huggingface discord channel)

3) To fix the above problem, refer -https://huggingface.co/datasets/John6666/forum2/blob/main/trl_kto_blow_up_memory_1.md


## ORPO Trainer

If you are facing a zip error, please try updating all of the libraries. There was a bug related to strict parameter for zip func, which is fixed


## GRPO Trainer

Zero Loss shown by GRPO is not a problem. Use Wandb to record metrics. If it shows gradient , then it is fine. 

Refer:

-> https://huggingface.co/datasets/John6666/forum3/blob/main/grpo_0_loss_issue_1.md

-> https://huggingface.co/datasets/John6666/forum3/blob/main/grpo_0_loss_issue_2.md
