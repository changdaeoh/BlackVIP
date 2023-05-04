#!/bin/bash
cd ../..

DATA=/YOURPATH
TRAINER=ZeroshotCLIP

SEED=1
SHOTS=16
# for CFG in rn50 rn101 vit_b32 vit_b16
for DATASET in eurosat svhn oxford_pets clevr
do
for CFG in vit_b16
do

DIR=output/${DATASET}/ZS/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

CUDA_VISIBLE_DEVICES=2 python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir ${DIR} \
--eval-only \
DATASET.SUBSAMPLE_CLASSES all

done
done