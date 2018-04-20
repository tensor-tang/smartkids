#!/bin/bash

#/usr/local/cuda/lib64:/usr/local/cuda-8.0/lib64:/usr/local/lib

#rm -rf models

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
python sks_main.py \
  --no_debug \
  --dataset 'imagenet' \
  --kids 3 10 \
  --num_epochs 1 \
  --learning_rate 0.05 \
  --learning_rate_decay_rate 0.9 \
  --batch_size 256 \
  --val_batch_size 256 \
  --data_format 'nchw' \
  --log_every_n_iters 10 \
  --val_every_n_epoch 0.01 \
  --test_every_n_epoch 0.1 \
  --ckpt_every_n_epoch 1.0 \
  --profil_every_n_epoch 2.0 \
  --summary_every_n_epoch 1.0 2>&1 | tee sks_imagenet.log
