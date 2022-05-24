from solo.attack.attack import AttackGenerate

from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torchvision
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from solo.methods.base import BaseMethod
from solo.utils.lars import LARSWrapper
from solo.utils.metrics import accuracy_at_k, weighted_mean
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader


class Training(pl.LightningModule):
    _SUPPORTED_ATTACKNET = {
        "resnet18": torchvision.models.resnet18(pretrained=True),
        "resnet50": torchvision.models.resnet50(pretrained=True),
    }

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        max_epochs: int,
        batch_size: int,
        optimizer: str,
        lars: bool,
        lr: float,
        weight_decay: float,
        exclude_bias_n_norm: bool,
        extra_optimizer_args: dict,
        scheduler: str,
        min_lr: float,
        warmup_start_lr: float,
        warmup_epochs: float,
        lr_decay_steps: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        """Implements linear evaluation.

        Args:
            backbone (nn.Module): backbone architecture for feature extraction.
            num_classes (int): number of classes in the dataset.
            max_epochs (int): total number of epochs.
            batch_size (int): batch size.
            optimizer (str): optimizer to use.
            lars (bool): whether to use lars or not.
            lr (float): learning rate.
            weight_decay (float): weight decay.
            exclude_bias_n_norm (bool): whether to exclude bias and batch norm from weight decay
                and lars adaptation.
            extra_optimizer_args (dict): extra optimizer arguments.
            scheduler (str): learning rate scheduler.
            min_lr (float): minimum learning rate for warmup scheduler.
            warmup_start_lr (float): initial learning rate for warmup scheduler.
            warmup_epochs (float): number of warmup epochs.
            lr_decay_steps (Optional[Sequence[int]], optional): list of epochs where the learning
                rate will be decreased. Defaults to None.
        """

        super().__init__()
        
        # classifer 跟 backbone 合併
        self.backbone = backbone
        # if hasattr(self.backbone, "inplanes"):
        #     features_dim = self.backbone.inplanes
        # else:
        #     features_dim = self.backbone.num_features
        # self.classifier = nn.Linear(features_dim, num_classes)  # type: ignore

        # training related
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lars = lars
        self.lr = lr
        self.weight_decay = weight_decay
        self.exclude_bias_n_norm = exclude_bias_n_norm
        self.extra_optimizer_args = extra_optimizer_args
        self.scheduler = scheduler
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.warmup_epochs = warmup_epochs
        self.lr_decay_steps = lr_decay_steps
        self._num_training_steps = None

        # all the other parameters
        self.extra_args = kwargs

        # #改true
        # for param in self.backbone.parameters():
        #     param.requires_grad = False 
        
        # Attack initial
        if self.extra_args['dataset']=='adv_imagenet':
            for net in set(kwargs['target_net']):
                if net == 'resnet18':
                    self.attack_resnet18 = self._SUPPORTED_ATTACKNET[net]
                if net == 'resnet50':
                    self.attack_resnet50 = self._SUPPORTED_ATTACKNET[net]
            

            if len(kwargs['attack_method']) > 1: #多個t，各產生一張圖
                self.attacks = [
                    self.prepare_attack(kwargs['dataset'], **attack_kwargs) for attack_kwargs in kwargs['attack_kwargs']
                ]
            else: #一個t，產生多張
                self.attacks = [self.prepare_attack(kwargs['dataset'], **kwargs['attack_kwargs'])]


    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds basic linear arguments.

        Args:
            parent_parser (ArgumentParser): argument parser that is used to create a
                argument group.

        Returns:
            ArgumentParser: same as the argument, used to avoid errors.
        """

        parser = parent_parser.add_argument_group("linear")

        # backbone args
        parser.add_argument("--backbone", choices=BaseMethod._SUPPORTED_BACKBONES, type=str)
        parser.add_argument("--pretrained", action="store_true")
        # for ViT
        parser.add_argument("--patch_size", type=int, default=16)

        # general train
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--lr", type=float, default=0.3)
        # parser.add_argument("--classifier_lr", type=float, default=0.3)
        parser.add_argument("--weight_decay", type=float, default=0.0001)
        parser.add_argument("--num_workers", type=int, default=4)

        # wandb
        parser.add_argument("--name")
        parser.add_argument("--project")
        parser.add_argument("--entity", default=None, type=str)
        parser.add_argument("--wandb", action="store_true")
        parser.add_argument("--offline", action="store_true")

        # optimizer
        SUPPORTED_OPTIMIZERS = ["sgd", "adam"]

        parser.add_argument("--optimizer", choices=SUPPORTED_OPTIMIZERS, type=str, required=True)
        parser.add_argument("--lars", action="store_true")
        parser.add_argument("--exclude_bias_n_norm", action="store_true")

        # scheduler
        SUPPORTED_SCHEDULERS = [
            "reduce",
            "warmup_cosine",
            "step",
            "exponential",
            "none",
        ]

        parser.add_argument("--scheduler", choices=SUPPORTED_SCHEDULERS, type=str, default="reduce")
        parser.add_argument("--lr_decay_steps", default=None, type=int, nargs="+")
        parser.add_argument("--min_lr", default=0.0, type=float)
        parser.add_argument("--warmup_start_lr", default=0.003, type=float)
        parser.add_argument("--warmup_epochs", default=10, type=int)

        return parent_parser

    def set_loaders(self, train_loader: DataLoader = None, val_loader: DataLoader = None) -> None:
        """Sets dataloaders so that you can obtain extra information about them.
        We currently only use to obtain the number of training steps per epoch.

        Args:
            train_loader (DataLoader, optional): training dataloader.
            val_loader (DataLoader, optional): validation dataloader.

        """

        if train_loader is not None:
            self.train_dataloader = lambda: train_loader

        if val_loader is not None:
            self.val_dataloader = lambda: val_loader

    @property
    def num_training_steps(self) -> int:
        """Compute the number of training steps for each epoch."""

        if self._num_training_steps is None:
            if self.trainer.train_dataloader is None:
                try:
                    dataloader = self.train_dataloader()
                except NotImplementedError:
                    raise RuntimeError(
                        "To use linear warmup cosine annealing lr"
                        "set the dataloader with .set_loaders(...)"
                    )

            dataset_size = getattr(self, "dali_epoch_size", None) or len(dataloader.dataset)

            dataset_size = self.trainer.limit_train_batches * dataset_size

            num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)

            if self.trainer.tpu_cores:
                num_devices = max(num_devices, self.trainer.tpu_cores)

            effective_batch_size = (
                self.batch_size * self.trainer.accumulate_grad_batches * num_devices
            )
            self._num_training_steps = dataset_size // effective_batch_size

        return self._num_training_steps

    def forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs forward pass of the frozen backbone and the linear layer for evaluation.

        Args:
            X (torch.tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing features and logits.
        """

        #去除no_grad
        # classifer 跟 backbone 合併
        # with torch.no_grad():   
        #     feats = self.backbone(X)
        # logits = self.classifier(feats)
        logits = self.backbone(X)
        return {"logits": logits}

    def configure_optimizers(self) -> Tuple[List, List]:
        """Configures the optimizer for the linear layer.

        Raises:
            ValueError: if the optimizer is not in (sgd, adam).
            ValueError: if the scheduler is not in not in (warmup_cosine, cosine, reduce, step,
                exponential).

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam
        else:
            raise ValueError(f"{self.optimizer} not in (sgd, adam)")

        optimizer = optimizer(
            self.backbone.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )

        if self.lars:
            optimizer = LARSWrapper(optimizer, exclude_bias_n_norm=self.exclude_bias_n_norm)

        # select scheduler
        if self.scheduler == "none":
            return optimizer

        if self.scheduler == "warmup_cosine":
            scheduler = {
                "scheduler": LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=self.warmup_epochs * self.num_training_steps,
                    max_epochs=self.max_epochs * self.num_training_steps,
                    warmup_start_lr=self.warmup_start_lr,
                    eta_min=self.min_lr,
                ),
                "interval": "step",
                "frequency": 1,
            }
        elif self.scheduler == "reduce":
            scheduler = ReduceLROnPlateau(optimizer)
        elif self.scheduler == "step":
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps, gamma=0.1)
        elif self.scheduler == "exponential":
            scheduler = ExponentialLR(optimizer, self.weight_decay)
        else:
            raise ValueError(
                f"{self.scheduler} not in (warmup_cosine, cosine, reduce, step, exponential)"
            )

        return [optimizer], [scheduler]

    def shared_step(
        self, batch: Tuple, batch_idx: int
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs operations that are shared between the training nd validation steps.

        Args:
            batch (Tuple): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
                batch size, loss, accuracy @1 and accuracy @5.
        """
        if self.extra_args['dataset']=='adv_imagenet':
            X, target, _= batch
        else:
            X, target = batch

        batch_size = X.size(0)

        out = self(X)["logits"]

        loss = F.cross_entropy(out, target)

        acc1, acc5 = accuracy_at_k(out, target, top_k=(1, 5))
        return batch_size, loss, acc1, acc5

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Performs the training step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            torch.Tensor: cross-entropy loss between the predictions and the ground truth.
        """

        # set backbone to eval mode
        # self.backbone.eval()

        # add attack
        # print(len(batch))
        # print("Before attack", batch[0].shape, batch[1].shape)
        if self.extra_args['dataset']=='adv_imagenet':
            for attack in self.attacks:
                x, targets, target_labels = batch
                adv_x = attack.attack.forward(x, targets, target_labels)
                batch[0] = torch.cat((batch[0], adv_x), 0)
                batch[1] = torch.cat((batch[1], batch[1]), 0)
            # print("After attack", batch[0].shape, batch[1].shape)

        _, loss, acc1, acc5 = self.shared_step(batch, batch_idx)

        log = {"train_loss": loss, "train_acc1": acc1, "train_acc5": acc5}
        self.log_dict(log, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, Any]:
        """Performs the validation step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            Dict[str, Any]:
                dict with the batch_size (used for averaging),
                the classification loss and accuracies.
        """

        batch_size, loss, acc1, acc5 = self.shared_step(batch, batch_idx)

        results = {
            "batch_size": batch_size,
            "val_loss": loss,
            "val_acc1": acc1,
            "val_acc5": acc5,
        }
        return results

    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.

        Args:
            outs (List[Dict[str, Any]]): list of outputs of the validation step.
        """

        val_loss = weighted_mean(outs, "val_loss", "batch_size")
        val_acc1 = weighted_mean(outs, "val_acc1", "batch_size")
        val_acc5 = weighted_mean(outs, "val_acc5", "batch_size")

        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}
        self.log_dict(log, sync_dist=True)

    def prepare_attack(self, dataset: str, **kwarg):
        """Prepares transforms for a specific dataset. Optionally uses multi crop.

        Args:
            dataset (str): name of the dataset.

        Returns:
            Any: a transformation for a specific dataset.
        """
        #print(kwarg)
        net_name = kwarg['target_net']
        if net_name == 'resnet18':
            attack_net = self.attack_resnet18 
        if net_name == 'resnet50':
            attack_net = self.attack_resnet50
        else:
            pass

        if dataset == "adv_imagenet":
            return AttackGenerate(data_name='imagenet', attack_net=attack_net, **kwarg)
        elif dataset == "adv_cifar10":
            return AttackGenerate(data_name='imagenet',  attack_net=attack_net,**kwarg)
        else:
            raise ValueError(f"{dataset} is not currently supported.")

