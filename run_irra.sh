#!/bin/bash
DATASET_NAME="CUHK-PEDES"

python train.py \
--name iira \
--batch_size 64 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+mlm+id' \
--num_epoch 60