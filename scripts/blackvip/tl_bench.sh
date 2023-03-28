#!/bin/bash
cd ../..
DATA=/YOURPATH
DATASET=$1
TRAINER=BLACKVIP
SHOTS=16
CFG=$2
ep=2

ptb=vit-mae-base
spsa_os=1.0

b1=0.9
alpha=0.4
gamma=$3
spsa_a=0.01
spsa_c=$4
p_eps=$5

opt_type='spsa-gc'

for SEED in 1
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}/shot${SHOTS}_ep${ep}/${opt_type}_b1${b1}/a${alpha}_g${gamma}_sa${spsa_a}_sc${spsa_c}_eps${p_eps}/seed${SEED}
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
    TRAINER.BLACKVIP.P_EPS $p_eps
    #fi
done