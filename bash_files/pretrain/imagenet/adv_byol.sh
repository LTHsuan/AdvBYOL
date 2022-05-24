python3 ../../../main_pretrain.py \
    --dataset adv_imagenet \
    --backbone resnet50 \
    --pretrained \
    --data_dir /data2 \
    --train_dir 1K_New/train \
    --val_dir 1K_New/val \
    --subset_class_num 1000 \
    --max_epochs 201 \
    --gpus 4,5,6,7 \
    --accelerator gpu \
    --strategy ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --eta_lars 0.001 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --accumulate_grad_batches 2 \
    --classifier_lr 0.2 \
    --weight_decay 1e-6 \
    --batch_size 400 \
    --num_workers 40 \
    --method Adv_byol \
    --num_crops_per_aug 1 1\
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 1.0 0.1 \
    --solarization_prob 0.0 0.2 \
    --attack_method fgsm \
    --target_net resnet50 \
    --distance Linf \
    --target False\
    --wandb \
    --name fgsmlinf-byol-Pretrainedresnet50-imagenet-200ep-v2 \
    --entity tinghsuan \
    --project solo-learn \
    --save_checkpoint \
    --checkpoint_dir ../../../experiment_result \
    --checkpoint_frequency 50 \
    --keep_previous_checkpoints \
    --proj_output_dim 256 \
    --proj_hidden_dim 4096 \
    --pred_hidden_dim 4096 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 1.0 \
    --momentum_classifier \
    #--dali \
    #--pretrained \
    #--subset_class_num 100 \
    # --eps 3 \
    # --stepsize 0.075 \
    # --auto_resume \
    # --resume_from_checkpoint \
    # --resume_from_checkpoint ../../../experiment_result/Adv_byol/2hme7aoy/fgsmlinf-byol-Pretrainedresnet50-imagenet-50ep-v2-2hme7aoy-ep=49.ckpt \
