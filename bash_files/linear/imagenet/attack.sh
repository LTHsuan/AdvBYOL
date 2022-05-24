python3 ../../../main_attack.py \
    --dataset adv_imagenet \
    --backbone resnet50 \
    --pretrained_ckpt ../../../experiment_result/linear/2r1cxpvi/fgsmlinf-byol-resnet50-imagenet100-700ep-linear-eval-2r1cxpvi-ep=79.ckpt\
    --data_dir /data2 \
    --train_dir 1K_New/train \
    --val_dir 1K_New/val \
    --subset_class_num 100 \
    --optimizer sgd \
    --gpus 3 \
    --batch_size 400 \
    --num_workers 40 \
    --attack_method pgd \
    --target_net resnet50 \
    --distance Linf \
    --target False \
    --wandb \
    --name fgsmlinf-byol-resnet50-imagenet100-700ep-pgd_linf_attack \
    --entity tinghsuan \
    --project solo-learn-attack \
    --save_checkpoint \
    --checkpoint_dir ../../../experiment_result \

    #--dali \
    #--pretrained_ckpt ../../../experiment_result/linear/3q8ut7l7/fgsml2-byol-resnet50-imagenet100-v2-linear-eval-3q8ut7l7-ep=79.ckpt \
    #--pretrained \
    #--subset_class_num 100 \
    # --eps 3 \
    # --stepsize 0.075 \
    
    #Black box ATTACK
    # --eps 16/255 \
    # --stepsize 16/25500 \

