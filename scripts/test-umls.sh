#!/bin/bash

python train.py link_prediction with \
dataset='umls' \
inductive=False \
dim=128 \
model='bert-bow' \
rel_model='transe' \
loss_fn='margin' \
encoder_name='bert-base-cased' \
regularizer=1e-2 \
max_len=32 \
num_negatives=64 \
lr=1e-3 \
use_scheduler=False \
batch_size=64 \
emb_batch_size=512 \
eval_batch_size=64 \
max_epochs=5 \
checkpoint=None \
use_cached_text=False
