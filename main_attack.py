from solo.attack.attack import AttackGenerate

from solo.args.setup import parse_args_attack
from solo.methods.base import BaseMethod

from solo.utils.classification_dataloader import prepare_data
from solo.utils.checkpointer import Checkpointer

import os
import numpy as np
import time
import wandb
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50

def prepare_attack(dataset, net, **kwarg):
    if dataset == "adv_imagenet":
        return AttackGenerate(data_name='imagenet', attack_net=net, **kwarg)
    elif dataset == "adv_cifar10":
        return AttackGenerate(data_name='cifar10',  attack_net=net,**kwarg)
    else:
        raise ValueError(f"{dataset} is not currently supported.")

def evaluate(args, attack, net, dataloader, wandb, device):
    net.eval()
    success_num = 0
    test_num= 0
    attack_success_num = 0
    num = 0

    for i, (image, labels, t_label) in enumerate(dataloader, 1):
        start_time=time.time()

        # generate adversary
        batchsize = image.shape[0]
        image, labels = image.to(device), labels.to(device)
        target_labels=None

        if attack.target:
            target_labels=t_label.to(device)
        
        # adv_image= attack.attack.forward(image, labels, target_labels, device)
        adv_image= attack.attack.forward(image, labels, target_labels)
        
        with torch.no_grad():
            # test clean acc and asr
            out = net(image)

            out_adv = net(adv_image)

            out_adv = torch.argmax(out_adv, dim=1)
            out = torch.argmax(out, dim=1)
            # print("out_adv", out_adv)
            # print("out", out)
            # print("labels", labels)
            
            test_num += (out == labels).sum()
            if attack.target:
                success_num +=(out_adv == target_labels).sum()

            else:
                success_num +=(out_adv == labels).sum()
                attack_success_num +=((out_adv != out) & (labels==out)).sum()

            num += batchsize
            test_acc = test_num.item() / num
            adv_acc = success_num.item() / num #given attack image model acc
            attack_success_acc = attack_success_num.item() / test_num.item()
            print("num", num)
            print("test", test_num)
            print("success", success_num)
            print("attack", attack_success_num)

            if i % 10 == 0:
                print("Target model %s, Dataset %s, epoch %d, test acc %.2f %%" %(args.backbone, args.dataset, i, test_acc*100 ))
                print("Attack name %s, dataset %s, epoch %d, asr %.2f %%\n" %(args.attack_method, args.dataset, i, adv_acc*100))
            
            wandb.log({"original_acc":test_acc*100, "adv_acc":adv_acc*100, "attack_succ":attack_success_acc*100})

        end_time=time.time()
        print('Time of one epoch: %f' %(end_time-start_time))

    total_num = len(dataloader.dataset)

    final_test_acc = test_num.item() / total_num
    success_num = success_num.item() / total_num
    # print('Final: Target model %s, clean %.2f %%' %(args.backbone, test_acc*100))
    print("Final: Attack %s, dataset %s, asr %.2f %%" %(args.attack_method, args.dataset, success_num*100))
    print("Final: Dataset %s, test acc %.2f %%" %(args.dataset, final_test_acc*100))
    


def main():
    args = parse_args_attack()

    callbacks = []
    # wandb logging
    if args.wandb:
        wandb.init(name=args.name, project=args.project, entity=args.entity, config=args)

    if args.save_checkpoint:
        # save checkpoint on last epoch only
        ckpt = Checkpointer(
            args,
            logdir=os.path.join(args.checkpoint_dir, "attack"),
            frequency=args.checkpoint_frequency,
        )
        callbacks.append(ckpt)

    assert args.backbone in BaseMethod._SUPPORTED_BACKBONES
    backbone_model = {
        "resnet18": resnet18,
        "resnet50": resnet50,
    }[args.backbone]

    if args.pretrained:
        backbone = backbone_model(args.pretrained)
    
    else:
        backbone = backbone_model()
        # backbone.fc = nn.Linear(2048, 100)
        assert (
        args.pretrained_ckpt.endswith(".ckpt")
        or args.pretrained_ckpt.endswith(".pth")
        or args.pretrained_ckpt.endswith(".pt")
        )
        ckpt_path = args.pretrained_ckpt
        state = torch.load(ckpt_path)["state_dict"]
        # print(state.keys())
        for k in list(state.keys()):
            if "encoder" in k:
                raise Exception(
                    "You are using an older checkpoint."
                    "Either use a new one, or convert it by replacing"
                    "all 'encoder' occurrences in state_dict with 'backbone'"
                )
            if "backbone" in k:
                state[k.replace("backbone.", "")] = state[k]

            if "classifier" in k:
                print(k)
                state[k.replace("classifier.", "fc.")] = state[k]

            del state[k]
        backbone.load_state_dict(state, strict=False)

    device = torch.device(args.gpus[0])
    
    backbone.to(device)

    if args.wandb:
        wandb.watch(backbone, log_freq=100)

    _ , val_loader = prepare_data(
        args.dataset,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset_class_num=args.subset_class_num,
    )

    attack_kwargs = args.attack_kwargs
    del attack_kwargs["target_net"]

    print(attack_kwargs)
    # print(backbone)
    print("prepare attack method")

    if len(args.attack_method) > 1: #多個t，各產生一張圖
        attacks = [
            prepare_attack(args.dataset, net=backbone, **attack_kwarg) for attack_kwarg in attack_kwargs
        ]
    else: #一個t，產生多張
        attacks = [prepare_attack(args.dataset, net=backbone, **attack_kwargs)]

    print("Total Attacks = {}".format(len(attacks)))

    for attack in attacks: 
        print("start to eval")
        evaluate(args, attack, backbone, val_loader ,wandb, device)



if __name__ == "__main__":
    main()
