#!/bin/bash

python train.py link_prediction with \
dataset='FB15k-237' \
inductive=True \
dim=128 \
model='blp' \
rel_model='distmult' \
loss_fn='margin' \
encoder_name='bert-base-cased' \
regularizer=0 \
max_len=32 \
num_negatives=64 \
lr=2e-5 \
use_scheduler=True \
batch_size=64 \
emb_batch_size=512 \
eval_batch_size=64 \
max_epochs=0 \
checkpoint="output/model-298.pt" \
use_cached_text=True
