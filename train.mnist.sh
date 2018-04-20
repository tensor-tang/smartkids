#!/bin/bash

#/usr/local/cuda/lib64:/usr/local/cuda-8.0/lib64:/usr/local/lib

#rm -rf models

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
python sks_main.py \
  --no_debug \
  --kids 3 10 \
  --dataset 'mnist' \
  --num_epochs 11 \
  --learning_rate 0.1 \
  --learning_rate_decay_rate 0.95 \
  --batch_size 1024 \
  --data_format 'nchw' \
  --log_every_n_iters 200 \
  --val_every_n_epoch 0.5 \
  --test_every_n_epoch 5.0 \
  --ckpt_every_n_epoch 10.0 \
  --profil_every_n_epoch 2.0 \
  --summary_every_n_epoch 1.0 2>&1 | tee sks_train.log
