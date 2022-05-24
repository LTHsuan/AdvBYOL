python3 ../../../main_AT.py \
    --dataset adv_imagenet \
    --backbone resnet50 \
    --pretrained \
    --data_dir /data2 \
    --train_dir 1K_New/train \
    --val_dir 1K_New/val \
    --subset_class_num 100 \
    --max_epochs 100 \
    --gpus 6,7 \
    --accelerator gpu \
    --strategy ddp \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.5 \
    --weight_decay 1e-6  \
    --batch_size 512 \
    --num_workers 20 \
    --method train \
    --attack_method fgsm \
    --target_net resnet50 \
    --distance L2 \
    --eps 3 \
    --stepsize 0.075 \
    --target False\
    --name fgsml2-AT-Pretrainedresnet50-imagenet100-100ep-v2\
    --entity tinghsuan \
    --project solo-learn \
    --wandb \
    --save_checkpoint \
    --checkpoint_dir ../../../experiment_result \
    # --dali \

    # --pretrained \

    ## if using adv_imagenet add attack config
    # --attack_method fgsm \
    # --target_net resnet50 \
    # --distance Linf \
    # --target False\
    # --eps 3 \
    # --stepsize 0.075 \
