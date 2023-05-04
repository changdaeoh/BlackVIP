#!/bin/bash
cd ../..
DATA=/YOURPATH
TRAINER=VPWB

CFG=vit_b16
SHOTS=16

DATASET=$1
ep=$2
lr=$3

for SEED in 1 2 3
do
    DIR=output/${DATASET}/shot${SHOTS}_ep${ep}/${TRAINER}/${CFG}/lr${lr}/seed${SEED}

    CUDA_VISIBLE_DEVICES=0 python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    OPTIM.LR $lr \
    OPTIM.MAX_EPOCH $ep \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES all
done