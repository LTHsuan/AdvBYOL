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



