#!/bin/bash
cd ../..
DATA=/YOURPATH
TRAINER=VPOUR
SHOTS=16
CFG=vit_b16

DATASET=$1
ep=$2

spsa_os=1.0
alpha=0.4
gamma=0.1

b1=$3
spsa_a=$4
spsa_c=$5

opt_type='spsa-gc'

for SEED in 1 2 3
do
    DIR=output/${DATASET}/${TRAINER}/${ptb}_${CFG}/shot${SHOTS}_ep${ep}/${opt_type}_b1${b1}/a${alpha}_g${gamma}_sa${spsa_a}_sc${spsa_c}/seed${SEED}
    CUDA_VISIBLE_DEVICES=6 python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    TRAIN.CHECKPOINT_FREQ 500 \
    TRAINER.VPOUR.SPSA_PARAMS [$spsa_os,$spsa_c,$spsa_a,$alpha,$gamma] \
    TRAINER.VPOUR.OPT_TYPE $opt_type \
    TRAINER.VPOUR.MOMS $b1 \
    OPTIM.MAX_EPOCH $ep \
    DATASET.SUBSAMPLE_CLASSES all
done