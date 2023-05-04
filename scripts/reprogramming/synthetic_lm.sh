#!/bin/bash
cd ../..
DATA=/YOURPATH
TRAINER=BAR
DATASET='locmnist'

CFG=vit_b16_noaug
SHOTS=16
ep=5000

fsize=$1
init_lr=$2
min_lr=$3

for SEED in 1 2 3
do
    DIR=output/${DATASET}_r1_f${fsize}/shot${SHOTS}_ep${ep}/${TRAINER}/${CFG}/lr${lr}/seed${SEED}

    CUDA_VISIBLE_DEVICES=0 python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    OPTIM.MAX_EPOCH $ep \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES all \
    TRAINER.BAR.LRS [$init_lr,$min_lr] \
    INPUT.SIZE [194,194] \
    DATASET.LOCMNIST.F_SIZE $fsize
done