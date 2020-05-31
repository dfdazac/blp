#!/bin/bash

python train.py link_prediction with \
dataset='Wikidata5M' \
inductive=True \
model='glove-dkrl' \
rel_model='transe' \
loss_fn='margin' \
regularizer=1e-3 \
max_len=64 \
num_negatives=64 \
lr=1e-4 \
use_scheduler=False \
batch_size=1024 \
emb_batch_size=12288 \
eval_batch_size=2 \
max_epochs=10 \
checkpoint=None \
use_cached_text=False
