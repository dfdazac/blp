#!/bin/bash

python train.py link_prediction with \
dataset='Wikidata5M' \
inductive=True \
dim=128 \
model='blp' \
rel_model='distmult' \
loss_fn='margin' \
encoder_name='bert-base-cased' \
regularizer=0 \
max_len=64 \
num_negatives=64 \
lr=5e-5 \
use_scheduler=True \
batch_size=1024 \
emb_batch_size=12288 \
eval_batch_size=2 \
max_epochs=0 \
checkpoint="output/model-341.pt" \
use_cached_text=True
