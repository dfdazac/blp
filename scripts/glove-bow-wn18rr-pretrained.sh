#!/bin/bash

python train.py link_prediction -u with \
dataset='WN18RR' \
inductive=True \
model='glove-bow' \
rel_model='transe' \
loss_fn='margin' \
regularizer=1e-2 \
max_len=32 \
num_negatives=64 \
lr=1e-3 \
use_scheduler=False \
batch_size=64 \
emb_batch_size=512 \
eval_batch_size=32 \
max_epochs=0 \
checkpoint="output/model-219.pt" \
use_cached_text=True
