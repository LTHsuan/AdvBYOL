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

from argparse import ArgumentParser
from pathlib import Path
import numpy as np


def dataset_args(parser: ArgumentParser):
    """Adds dataset-related arguments to a parser.

    Args:
        parser (ArgumentParser): parser to add dataset args to.
    """

    SUPPORTED_DATASETS = [
        "cifar10",
        "cifar100",
        "stl10",
        "imagenet",
        "imagenet100",
        "custom",
        "adv_imagenet",
        "adv_cifar10"
    ]

    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, type=str, required=True)

    # dataset path
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--train_dir", type=Path, default=None)
    parser.add_argument("--val_dir", type=Path, default=None)
    parser.add_argument("--subset_class_num", type=int, default=None)

    # dali (imagenet-100/imagenet/custom only)
    parser.add_argument("--dali", action="store_true")
    parser.add_argument("--dali_device", type=str, default="gpu")


def augmentations_args(parser: ArgumentParser):
    """Adds augmentation-related arguments to a parser.

    Args:
        parser (ArgumentParser): parser to add augmentation args to.
    """

    # cropping
    parser.add_argument("--num_crops_per_aug", type=int, default=[2], nargs="+")

    # color jitter
    parser.add_argument("--brightness", type=float, default=[0.4], nargs="+")
    parser.add_argument("--contrast", type=float, default=[0.4], nargs="+")
    parser.add_argument("--saturation", type=float, default=[0.2], nargs="+")
    parser.add_argument("--hue", type=float, default=[0.1], nargs="+")
    parser.add_argument("--color_jitter_prob", type=float, default=[0.8], nargs="+")

    # other augmentation probabilities
    parser.add_argument("--gray_scale_prob", type=float, default=[0.2], nargs="+")
    parser.add_argument("--horizontal_flip_prob", type=float, default=[0.5], nargs="+")
    parser.add_argument("--gaussian_prob", type=float, default=[0.5], nargs="+")
    parser.add_argument("--solarization_prob", type=float, default=[0.0], nargs="+")

    # cropping
    parser.add_argument("--crop_size", type=int, default=[224], nargs="+")
    parser.add_argument("--min_scale", type=float, default=[0.08], nargs="+")
    parser.add_argument("--max_scale", type=float, default=[1.0], nargs="+")

    # debug
    parser.add_argument("--debug_augmentations", action="store_true")
    

def linear_augmentations_args(parser: ArgumentParser):
    parser.add_argument("--crop_size", type=int, default=[224], nargs="+")


def custom_dataset_args(parser: ArgumentParser):
    """Adds custom data-related arguments to a parser.

    Args:
        parser (ArgumentParser): parser to add augmentation args to.
    """

    # custom dataset only
    parser.add_argument("--no_labels", action="store_true")

    # for custom dataset
    parser.add_argument("--mean", type=float, default=[0.485, 0.456, 0.406], nargs="+")
    parser.add_argument("--std", type=float, default=[0.228, 0.224, 0.225], nargs="+")


def attack_augmentations_args(parser: ArgumentParser):
    """Adds attack arguments to a parser.

    Args:
        parser (ArgumentParser): parser to add attack augmentation args to.
    """

    # attack
    parser.add_argument("--attack_method", nargs="+")
    parser.add_argument("--target_net", nargs="+")
    parser.add_argument("--distance", default=['Linf'], choices=['Linf', 'L2'], nargs="+")
    parser.add_argument('--loss', default=['ce'], help= 'loss for fgsm, bim, pgd, mim, dim and tim', choices= ['ce', 'cw'], nargs="+")
    parser.add_argument('--target', type=eval, default=[False], help= 'target for attack', nargs="+", choices=[True, False])
    #BIM, PGD, MIM, DIM, FGSM
    parser.add_argument('--eps', type= float, default=[8/255], help='linf: 8/255.0 and l2: 3.0', nargs="+")
    parser.add_argument('--stepsize', type= float, default=[8/2550], help='alpha, linf: 8/2550.0 and l2: (2.5*eps)/steps that is 0.075', nargs="+")
    parser.add_argument('--steps', type= int, default=[20], help='linf: 100 and l2: 100, steps is set to 100 if attack is apgd', nargs="+")
    #DIM, MIM
    parser.add_argument('--decay_factor', type= float, default=[1.0], help='momentum is used', nargs="+")
    #DIM
    parser.add_argument('--resize_rate', type= float, default=[0.85], help='dim is used', nargs="+")    #0.9
    parser.add_argument('--diversity_prob', type= float, default=[0.7], help='dim is used', nargs="+")    #0.5
    #DeepFool
    parser.add_argument('--overshoot', type= float, default=[0.02], nargs="+")
    parser.add_argument('--max_iter', type= int, default=[50], nargs="+")
    #CW
    parser.add_argument('--kappa', type= float, default=[0.0], help='confidence of c&w', nargs="+")
    parser.add_argument('--cw_lr', type= float, default=[0.2], nargs="+")
    parser.add_argument('--init_const', type= float, default=[0.01], nargs="+")
    parser.add_argument('--binary_search_steps', type= int, default=[4], nargs="+")
    parser.add_argument('--max_cw_iter', type= int, default=[200], nargs="+")
    #NES
    parser.add_argument('--bt_max_queries', type= int, default=[20000], help='max_queries for black-box attack based on queries', nargs="+")
    parser.add_argument('--nes_samples', default= [10], help='nes_samples for nes', nargs="+")
    parser.add_argument('--nes_per_draw', type= int, default=[20], help='nes_iters for nes', nargs="+")