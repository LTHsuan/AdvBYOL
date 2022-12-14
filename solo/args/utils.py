# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
from argparse import Namespace
from contextlib import suppress


N_CLASSES_PER_DATASET = {
    "cifar10": 10,
    "cifar100": 100,
    "stl10": 10,
    "imagenet": 1000,
    "imagenet100": 100,
    "adv_imagenet": 1000,
}

def attack_setup_pretrain(args: Namespace):
    unique_augs = max(
        len(p)
        for p in [
            #Adding attack args
            args.attack_method,
            args.target_net,
            args.distance,
            args.loss,
            args.target,
            args.eps,                #bim, pgd, mim, dim
            args.stepsize,         #bim, pgd, mim, dim
            args.steps,              #bim, pgd, mim, dim
            args.decay_factor,       #mim, dim
            args.resize_rate,        #dim
            args.diversity_prob,     #dim
            args.overshoot,          #deepfool
            args.max_iter,          #deepfool
            args.max_cw_iter,           #cw
            args.binary_search_steps,    #cw
            args.cw_lr,                 #cw
            args.init_const,         #cw
            args.kappa,              #cw (confidence)
            args.bt_max_queries,
            args.nes_samples,
            args.nes_per_draw
        ]
    )
    # assert len(args.num_crops_per_aug) == unique_augs

    # assert that either all unique augmentation pipelines have a unique
    # parameter or that a single parameter is replicated to all pipelines
    for p in [
        "attack_method",
        "target_net",
        "distance",
        "loss",
        "target",
        "eps",                #bim, pgd, mim, dim
        "stepsize",         #bim, pgd, mim, dim
        "steps",              #bim, pgd, mim, dim
        "decay_factor",       #mim, dim
        "resize_rate",        #dim
        "diversity_prob",     #dim
        "overshoot",          #deepfool
        "max_iter",          #deepfool
        "max_cw_iter",           #cw
        "binary_search_steps",    #cw
        "cw_lr",                 #cw
        "init_const",         #cw
        "kappa",              #cw (confidence)
        "bt_max_queries",
        "nes_samples",
        "nes_per_draw",
    ]:
        values = getattr(args, p)
        n = len(values)
        assert n == unique_augs or n == 1

        if n == 1:
            setattr(args, p, getattr(args, p) * unique_augs)

    args.attack_unique_augs = unique_augs
    # print("attack_unique_augs=",unique_augs)

    if unique_augs > 1:
        args.attack_kwargs = [
            dict(
                attack_method=attack_method,
                target_net=target_net,
                distance=distance,
                loss=loss,
                target=target,
                eps=eps,
                stepsize=stepsize,
                steps=steps,
                decay_factor=decay_factor,
                resize_rate=resize_rate,
                diversity_prob=diversity_prob,
                overshoot=overshoot,
                max_iter=max_iter,
                max_cw_iter=max_cw_iter,
                binary_search_steps=binary_search_steps,
                cw_lr=cw_lr,
                init_const=init_const,
                kappa=kappa,
                bt_max_queries=bt_max_queries,
                nes_samples=nes_samples,
                nes_per_draw=nes_per_draw,
            )
            for (
                attack_method,
                target_net,
                distance,
                loss,
                target,
                eps,                #bim, pgd, mim, dim
                stepsize,         #bim, pgd, mim, dim
                steps,              #bim, pgd, mim, dim
                decay_factor,       #mim, dim
                resize_rate,        #dim
                diversity_prob,     #dim
                overshoot,          #deepfool
                max_iter,          #deepfool
                max_cw_iter,           #cw
                binary_search_steps,    #cw
                cw_lr,                 #cw
                init_const,         #cw
                kappa,              #cw (confidence)
                bt_max_queries,
                nes_samples,
                nes_per_draw
            ) in zip(
                args.attack_method,
                args.target_net,
                args.distance,
                args.loss,
                args.target,
                args.eps,                #bim, pgd, mim, dim
                args.stepsize,         #bim, pgd, mim, dim
                args.steps,              #bim, pgd, mim, dim
                args.decay_factor,       #mim, dim
                args.resize_rate,        #dim
                args.diversity_prob,     #dim
                args.overshoot,          #deepfool
                args.max_iter,          #deepfool
                args.max_cw_iter,           #cw
                args.binary_search_steps,    #cw
                args.cw_lr,                 #cw
                args.init_const,         #cw
                args.kappa,              #cw (confidence)
                args.bt_max_queries,
                args.nes_samples,
                args.nes_per_draw,
            )
        ]

        # find number of big/small crops
        # big_size = args.crop_size[0]
        # num_large_crops = num_small_crops = 0
        # for size, n_crops in zip(args.crop_size, args.num_crops_per_aug):
        #     if big_size == size:
        #         num_large_crops += n_crops
        #     else:
        #         num_small_crops += n_crops
        # args.num_large_crops = num_large_crops
        # args.num_small_crops = num_small_crops

    else:
        args.attack_kwargs = dict(
            attack_method=args.attack_method[0],
            target_net=args.target_net[0],
            distance=args.distance[0],
            loss=args.loss[0],
            target=args.target[0],
            eps=args.eps[0],
            stepsize=args.stepsize[0],
            steps=args.steps[0],
            decay_factor=args.decay_factor[0],
            resize_rate=args.resize_rate[0],
            diversity_prob=args.diversity_prob[0],
            overshoot=args.overshoot[0],
            max_iter=args.max_iter[0],
            max_cw_iter=args.max_cw_iter[0],
            binary_search_steps=args.binary_search_steps[0],
            cw_lr=args.cw_lr[0],
            init_const=args.init_const[0],
            kappa=args.kappa[0],
            bt_max_queries=args.bt_max_queries[0],
            nes_samples=args.nes_samples[0],
            nes_per_draw=args.nes_per_draw[0],
        )

        # # find number of big/small crops
        # args.num_large_crops = args.num_crops_per_aug[0]
        # args.num_small_crops = 0

def additional_setup_pretrain(args: Namespace):
    """Provides final setup for pretraining to non-user given parameters by changing args.

    Parsers arguments to extract the number of classes of a dataset, create
    transformations kwargs, correctly parse gpus, identify if a cifar dataset
    is being used and adjust the lr.

    Args:
        args (Namespace): object that needs to contain, at least:
        - dataset: dataset name.
        - brightness, contrast, saturation, hue, min_scale: required augmentations
            settings.
        - dali: flag to use dali.
        - optimizer: optimizer name being used.
        - gpus: list of gpus to use.
        - lr: learning rate.

        [optional]
        - gaussian_prob, solarization_prob: optional augmentations settings.
    """

    if args.dataset in N_CLASSES_PER_DATASET:
        args.num_classes = N_CLASSES_PER_DATASET[args.dataset] if args.subset_class_num is not None else args.subset_class_num
    else:
        # hack to maintain the current pipeline
        # even if the custom dataset doesn't have any labels
        dir_path = args.data_dir / args.train_dir
        args.num_classes = max(
            1,
            len([entry.name for entry in os.scandir(dir_path) if entry.is_dir]),
        )if args.subset_class_num is not None else args.subset_class_num

    unique_augs = max(
        len(p)
        for p in [
            args.brightness,
            args.contrast,
            args.saturation,
            args.hue,
            args.color_jitter_prob,
            args.gray_scale_prob,
            args.horizontal_flip_prob,
            args.gaussian_prob,
            args.solarization_prob,
            args.crop_size,
            args.min_scale,
            args.max_scale,
        ]
    )
    assert len(args.num_crops_per_aug) == unique_augs

    # assert that either all unique augmentation pipelines have a unique
    # parameter or that a single parameter is replicated to all pipelines
    for p in [
        "brightness",
        "contrast",
        "saturation",
        "hue",
        "color_jitter_prob",
        "gray_scale_prob",
        "horizontal_flip_prob",
        "gaussian_prob",
        "solarization_prob",
        "crop_size",
        "min_scale",
        "max_scale",
    ]:
        values = getattr(args, p)
        n = len(values)
        assert n == unique_augs or n == 1

        if n == 1:
            setattr(args, p, getattr(args, p) * unique_augs)

    args.unique_augs = unique_augs

    if unique_augs > 1:
        args.transform_kwargs = [
            dict(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
                color_jitter_prob=color_jitter_prob,
                gray_scale_prob=gray_scale_prob,
                horizontal_flip_prob=horizontal_flip_prob,
                gaussian_prob=gaussian_prob,
                solarization_prob=solarization_prob,
                crop_size=crop_size,
                min_scale=min_scale,
                max_scale=max_scale,
            )
            for (
                brightness,
                contrast,
                saturation,
                hue,
                color_jitter_prob,
                gray_scale_prob,
                horizontal_flip_prob,
                gaussian_prob,
                solarization_prob,
                crop_size,
                min_scale,
                max_scale,
            ) in zip(
                args.brightness,
                args.contrast,
                args.saturation,
                args.hue,
                args.color_jitter_prob,
                args.gray_scale_prob,
                args.horizontal_flip_prob,
                args.gaussian_prob,
                args.solarization_prob,
                args.crop_size,
                args.min_scale,
                args.max_scale,
            )
        ]

        # find number of big/small crops
        big_size = args.crop_size[0]
        num_large_crops = num_small_crops = 0
        for size, n_crops in zip(args.crop_size, args.num_crops_per_aug):
            if big_size == size:
                num_large_crops += n_crops
            else:
                num_small_crops += n_crops
        args.num_large_crops = num_large_crops
        args.num_small_crops = num_small_crops
    else:
        args.transform_kwargs = dict(
            brightness=args.brightness[0],
            contrast=args.contrast[0],
            saturation=args.saturation[0],
            hue=args.hue[0],
            color_jitter_prob=args.color_jitter_prob[0],
            gray_scale_prob=args.gray_scale_prob[0],
            horizontal_flip_prob=args.horizontal_flip_prob[0],
            gaussian_prob=args.gaussian_prob[0],
            solarization_prob=args.solarization_prob[0],
            crop_size=args.crop_size[0],
            min_scale=args.min_scale[0],
            max_scale=args.max_scale[0],
        )

        # find number of big/small crops
        args.num_large_crops = args.num_crops_per_aug[0]
        args.num_small_crops = 0

 
    # add support for custom mean and std
    if args.dataset == "custom":
        if isinstance(args.transform_kwargs, dict):
            args.transform_kwargs["mean"] = args.mean
            args.transform_kwargs["std"] = args.std
        else:
            for kwargs in args.transform_kwargs:
                kwargs["mean"] = args.mean
                kwargs["std"] = args.std

    # create backbone-specific arguments
    args.backbone_args = {"cifar": args.dataset in ["cifar10", "cifar100"]}
    if args.dataset in ["imagenet", "imagenet100", "adv_imagenet"]:
        args.backbone_args["pretrained"] = args.pretrained

    if "resnet" in args.backbone:
        args.backbone_args["zero_init_residual"] = args.zero_init_residual
    elif "convnext" not in args.backbone:
        # dataset related for all transformers
        crop_size = args.crop_size[0]
        args.backbone_args["img_size"] = crop_size
        if "vit" in args.backbone:
            args.backbone_args["patch_size"] = args.patch_size

    with suppress(AttributeError):
        del args.zero_init_residual
    with suppress(AttributeError):
        del args.patch_size

    if args.dali:
        assert args.dataset in ["imagenet100", "imagenet", "custom"]

    args.extra_optimizer_args = {}
    if args.optimizer == "sgd":
        args.extra_optimizer_args["momentum"] = 0.9

    if isinstance(args.gpus, int):
        args.gpus = [args.gpus]
    elif isinstance(args.gpus, str):
        args.gpus = [int(gpu) for gpu in args.gpus.split(",") if gpu]

    # adjust lr according to batch size
    args.lr = args.lr * args.batch_size * len(args.gpus) / 256

def additional_setup_train(args: Namespace):
    """Provides final setup for linear evaluation to non-user given parameters by changing args.

    Parsers arguments to extract the number of classes of a dataset, correctly parse gpus, identify
    if a cifar dataset is being used and adjust the lr.

    Args:
        args: Namespace object that needs to contain, at least:
        - dataset: dataset name.
        - optimizer: optimizer name being used.
        - gpus: list of gpus to use.
        - lr: learning rate.
    """

    if args.dataset in N_CLASSES_PER_DATASET:
        args.num_classes = N_CLASSES_PER_DATASET[args.dataset] if args.subset_class_num is not None else args.subset_class_num
    else:
        # hack to maintain the current pipeline
        # even if the custom dataset doesn't have any labels
        dir_path = args.data_dir / args.train_dir
        args.num_classes = max(
            1,
            len([entry.name for entry in os.scandir(dir_path) if entry.is_dir]),
        ) if args.subset_class_num is not None else args.subset_class_num

    # create backbone-specific arguments
    args.backbone_args = {"cifar": args.dataset in ["cifar10", "cifar100"]}

    if args.dataset in ["imagenet", "imagenet100", "adv_imagenet"]:
        args.backbone_args["pretrained"] = args.pretrained

    if "resnet" not in args.backbone and "convnext" not in args.backbone:
        # dataset related for all transformers
        crop_size = args.crop_size[0]
        args.backbone_args["img_size"] = crop_size
        if "vit" in args.backbone:
            args.backbone_args["patch_size"] = args.patch_size

    with suppress(AttributeError):
        del args.patch_size

    if args.dali:
        assert args.dataset in ["imagenet100", "imagenet", "custom"]

    args.extra_optimizer_args = {}
    if args.optimizer == "sgd":
        args.extra_optimizer_args["momentum"] = 0.9

    if isinstance(args.gpus, int):
        args.gpus = [args.gpus]
    elif isinstance(args.gpus, str):
        args.gpus = [int(gpu) for gpu in args.gpus.split(",") if gpu]

def additional_setup_linear(args: Namespace):
    """Provides final setup for linear evaluation to non-user given parameters by changing args.

    Parsers arguments to extract the number of classes of a dataset, correctly parse gpus, identify
    if a cifar dataset is being used and adjust the lr.

    Args:
        args: Namespace object that needs to contain, at least:
        - dataset: dataset name.
        - optimizer: optimizer name being used.
        - gpus: list of gpus to use.
        - lr: learning rate.
    """

    if args.dataset in N_CLASSES_PER_DATASET:
        args.num_classes = N_CLASSES_PER_DATASET[args.dataset] if args.subset_class_num is not None else args.subset_class_num
    else:
        # hack to maintain the current pipeline
        # even if the custom dataset doesn't have any labels
        dir_path = args.data_dir / args.train_dir
        args.num_classes = max(
            1,
            len([entry.name for entry in os.scandir(dir_path) if entry.is_dir]),
        ) if args.subset_class_num is not None else args.subset_class_num

    # create backbone-specific arguments
    args.backbone_args = {"cifar": args.dataset in ["cifar10", "cifar100"]}
    if "resnet" not in args.backbone and "convnext" not in args.backbone:
        # dataset related for all transformers
        crop_size = args.crop_size[0]
        args.backbone_args["img_size"] = crop_size
        if "vit" in args.backbone:
            args.backbone_args["patch_size"] = args.patch_size

    with suppress(AttributeError):
        del args.patch_size

    if args.dali:
        assert args.dataset in ["imagenet100", "imagenet", "custom"]

    args.extra_optimizer_args = {}
    if args.optimizer == "sgd":
        args.extra_optimizer_args["momentum"] = 0.9

    if isinstance(args.gpus, int):
        args.gpus = [args.gpus]
    elif isinstance(args.gpus, str):
        args.gpus = [int(gpu) for gpu in args.gpus.split(",") if gpu]
