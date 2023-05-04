#!/bin/bash
cd ../..
DATA=/YOURPATH
TRAINER=BLACKVIP
SHOTS=16
CFG=vit_b16_noaug
ptb=vit-mae-base

DATASET='locmnist'
ep=5000

spsa_os=1.0
gamma=0.1

fsize=$1
b1=$2
alpha=$3
spsa_a=$4
spsa_c=$5
p_eps=1.0

opt_type='spsa-gc'

for SEED in 1 2 3
do
    DIR=output/${DATASET}_r1_f${fsize}/${TRAINER}/${ptb}_${CFG}/shot${SHOTS}_ep${ep}/${opt_type}_b1${b1}/a${alpha}_g${gamma}_sa${spsa_a}_sc${spsa_c}_eps${p_eps}/seed${SEED}
    # if [ -d "$DIR" ]; then
    #     echo "Oops! The results exist at ${DIR} (so skip this job)"
    # else
    CUDA_VISIBLE_DEVICES=4 python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    TRAIN.CHECKPOINT_FREQ 500 \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES all \
    OPTIM.MAX_EPOCH $ep \
    TRAINER.BLACKVIP.PT_BACKBONE $ptb \
    TRAINER.BLACKVIP.SPSA_PARAMS [$spsa_os,$spsa_c,$spsa_a,$alpha,$gamma] \
    TRAINER.BLACKVIP.OPT_TYPE $opt_type \
    TRAINER.BLACKVIP.MOMS $b1 \
    TRAINER.BLACKVIP.P_EPS $p_eps \
    DATASET.LOCMNIST.F_SIZE $fsize
    #fi
done