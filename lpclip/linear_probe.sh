feature_dir=clip_feat

for DATASET in Resisc45
do
    python linear_probe.py \
    --dataset ${DATASET} \
    --feature_dir ${feature_dir} \
    --num_step 8 \
    --num_run 3 \
    --use_wandb \
    --backbone ViT-B/16 \
    --wandb_project 0129_lpreport
done
