![adv_SSL](https://user-images.githubusercontent.com/83267883/183443421-2f6186f3-5d53-4493-b988-007f9ecdbf30.png)

# AdvBYOL:Adversarial Bootstrap Your Own Latent for Model Robustness against Adversarial Attacks

This is a demo implementation of Adversarial Bootstrap Your Own Latent (AdvBYOL) to train the model in self-supervised learning, which learns to distinguish the adversarial example from the original data without providing the correct label. We validate our method on the large-scale ImageNet dataset, and obtains comparable robust accuracy over state-of-the-art supervised adversarial learning approaches.

---
## Requirements
* torch
* torchvision
* tqdm
* einops
* wandb
* pytorch-lightning
* lightning-bolts
* torchmetrics
* scipy
* timm

**Optional**:
* nvidia-dali
* matplotlib
* seaborn
* pandas
* umap-learn

---
## Preparation
Please download ImageNet1K dataset (https://www.image-net.org/download.php).Then unzip folder follow imageNet folder structure.

---
## Training
### Pretraining
- For quickly start, you can simply use the `AdvBYOL/bash_files/pretrain/imagenet/adv_byol.sh` to Pretrain 
- If you want to modify some hyper-parameters, please edit them in the configuration file `AdvBYOL/bash_files/pretrain/imagenet/adv_byol.sh` following the explanations below:
  - `dataset`:
  - `backbone`:
  - `pretrained`:
  - `data_dir`: 
  - `train_dir`:
  - `val_dir` :
  - `subset_class_num`:
  - `max_epochs`:
  - `gpus`:
  - `accelerator`:
  - `strategy`:
  - `sync_batchnorm`:
  - `precision`:
  - `optimizer`:
  - `lars`:
  - `eta_lars`:
  - `exclude_bias_n_norm`:
  - `scheduler`:
  - `lr`:
  - `accumulate_grad_batches`:
  - `classifier_lr`:
  - `weight_decay`:
  - `batch_size`:
  - `num_workers`:
  - `method`:
  - `num_crops_per_aug`:
  - `brightness`:
  - `contrast`:
  - `saturation`:
  - `hue`:
  - `gaussian_prob`:
  - `solarization_prob`:
  - `attack_method`:
  - `target_net`:
  - `distance`:
  - `target`:
  - `wandb`:
  - `name`:
  - `entity`:
  - `project`:
  - `save_checkpoint`:
  - `checkpoint_dir`:
  - `checkpoint_frequency`:
  - `keep_previous_checkpoints`:
  - `proj_output_dim`:
  - `proj_hidden_dim`:
  - `pred_hidden_dim`:
  - `base_tau_momentum`:
  - `final_tau_momentum`:
  - `momentum_classifier`:
  
### Linear Evaluation
- For quickly start, you can simply use the `AdvBYOL/bash_files/linear/imagenet/byol.sh` to Finetuned(Linear evaluation)
- 

### Attack Model (Test the Robustness of Model)
- For quickly start, you can simply use the `AdvBYOL/bash_files/linear/imagenet/attack.sh` to Attack the SSL Model(Test the Robustness of Model)
- 

## Acknowledgements
This code refers to the following projects:
1. SOLO-Learn GitHub (https://github.com/vturrisi/solo-learn)
2. Adversarial Attack Method (https://github.com/Xiang-cd/realsafe)
