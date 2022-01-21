#!/bin/bash

python train.py link_prediction with \
dataset='WN18RR' \
inductive=True \
model='bert-bow' \
rel_model='transe' \
loss_fn='margin' \
regularizer=1e-2 \
max_len=32 \
num_negatives=64 \
lr=1e-4 \
use_scheduler=False \
batch_size=64 \
emb_batch_size=512 \
eval_batch_size=16 \
max_epochs=0 \
checkpoint="output/model-220.pt" \
use_cached_text=True
