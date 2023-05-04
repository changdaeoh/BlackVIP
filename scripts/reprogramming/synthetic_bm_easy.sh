#!/bin/bash
cd ../..
DATA=/YOURPATH
TRAINER=BAR
DATASET='colour_biased_mnist'

CFG=vit_b16_noaug
SHOTS=16
ep=5000

init_lr=$1
min_lr=$2

for SEED in 1 2 3
do
    DIR=output/${DATASET}_easy/shot${SHOTS}_ep${ep}/${TRAINER}/${CFG}/lr${lr}/seed${SEED}

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
    DATASET.COLOUR_BIASED_MNIST.TRAIN_RHO 0.8 \
    DATASET.COLOUR_BIASED_MNIST.TEST_RHO 0.2
done