#!/bin/bash

cd ../..

# custom config
DATA=/YOURPATH
TRAINER=FTCLIP

CFG=vit_b16
SHOTS=16
EP=1000
WEP=100

for SEED in 1 2 3
do
for LR in 1e-5
do
for WD in 1e-2
do
for DATASET in clevr
do
    DIR=output/${DATASET}/shots${SHOTS}/${TRAINER}/${CFG}_EP${EP}_LR${LR}_WD${WD}/seed${SEED}
    # if [ -d "$DIR" ]; then
    #     echo "Oops! The results exist at ${DIR} (so skip this job)"
    # else
    CUDA_VISIBLE_DEVICES=6 python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    OPTIM.MAX_EPOCH ${EP} \
    OPTIM.WARMUP_EPOCH ${WEP} \
    OPTIM.WEIGHT_DECAY ${WD} \
    OPTIM.LR ${LR} \
    TRAIN.CHECKPOINT_FREQ 100 \
    TRAINER.FTCLIP.METHOD 'ft' \
    DATASET.SUBSAMPLE_CLASSES all
    # fi
done
done
done
done