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
- If you want to modify some hyper-parameters, please edit them in the configuration file `AdvBYOL/bash_files/pretrain/imagenet/adv_byol.sh` following the explanations below (More hyper-parameters are in the file `AdvBYOL/solo/args/dataset.py`):
  #### Training args.
  - `dataset`: Please set as 'adv_imagenet' which contain the attack method as data augmentation.
  - `backbone`: The model for SSL training
  - `pretrained`: Model with pretrained weight or not
  - `data_dir`: directory according to your path
  - `train_dir`: training dataset directory according to your path
  - `val_dir`: valudation dataset directory according to your path
  - `subset_class_num`: training using numbers of classes from dataset.
  - `max_epochs`: training expoch
  - `gpus`: CUDA_VISIBLE_DEVICES list
  - `accelerator`: GPU accelerate
  - `strategy`: set as ddp, ssing 'DISTRIBUTED DATA PARALLEL' to train
  - `sync_batchnorm`: Using batch normalization
  - `optimizer`: name of the optimizer.
  - `lars`: flag indicating if lars should be used.
  - `eta_lars`:eta parameter for lars.
  - `exclude_bias_n_norm`: flag indicating if bias and norms should be excluded from lars.
  - `scheduler`: name of the scheduler.
  - `lr`: Learning Rate for training
  - `accumulate_grad_batches`: number of batches for gradient accumulation.
  - `classifier_lr`: learning rate for the online linear classifier.
  - `weight_decay`: weight decay for optimizer.
  - `batch_size`: number of samples in the batch.
  - `num_workers`: numbers of workers
  - `method`: Please set as `Adv_byol` which contain the attack method as data augmentation.
  #### SSL data argumetation args.
  - `num_crops_per_aug`: data augmentation for SSL training 
  - `brightness`: (color jitter)Brightness probability
  - `contrast`: (color jitter)Contrast probability
  - `saturation`: (color jitter)Saturation probability
  - `hue`: (color jitter)Hue probability 
  - `gaussian_prob`: Gaussian Blur Probability
  - `solarization_prob`: solarization Probability
  #### Adversarial Attack args.
  - `attack_method`: Adversarial Attack method
  - `target_net`: Model which attack by attack method to generate the attack image for training
  - `distance`: distance to reduce the attack image and original image
  - `target`: Target attack or Non-target Attack
  #### Wandb & Checkpoint args.
  - `wandb`: Using wandb or not
  - `name`: name for saving this training in Wandb
  - `entity`: Wandb User name
  - `project`: Wandb project's name
  - `save_checkpoint`:flag indicating if save checkpoint should be used.
  - `checkpoint_dir`: the directory for saving checkpoint
  - `checkpoint_frequency`: how frequency to save checkpoint
  - `keep_previous_checkpoints`: flag indicating keeping previous checkpoint or not
  #### SSL Training method's args.
  - `proj_output_dim`: number of dimensions of projected features.
  - `proj_hidden_dim`: number of neurons of the hidden layers of the projector.
  - `pred_hidden_dim`: number of neurons of the hidden layers of the predictor.
  - `base_tau_momentum`: base value of the weighting decrease coefficient
  - `final_tau_momentum`: final value of the weighting decrease coefficient
  - `momentum_classifier`: whether or not to train a classifier on top of the momentum backbone.
  
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
