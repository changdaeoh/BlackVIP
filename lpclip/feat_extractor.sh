DATA=YOURPATH
OUTPUT='./clip_feat/'
#SEED=1

for SEED in 1 2 3
do
for DATASET in resisc45
do
    for SPLIT in train val test
    do
        CUDA_VISIBLE_DEVICES=6 python feat_extractor.py \
        --split ${SPLIT} \
        --root ${DATA} \
        --seed ${SEED} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/CoOp/vit16_val.yaml \
        --output-dir ${OUTPUT} \
        --eval-only \
        DATASET.NUM_SHOTS 16 \
        DATASET.SUBSAMPLE_CLASSES all
    done
done
done