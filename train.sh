#!/usr/bin/env bash
nohup python -m src.train --dataset davis \
--atom_hidden 512 1024   --pair_hidden 512 1024  --graph_features 1024 \
--num_filters 32 64 96 --filters_length 4 12 8 \
--biInteraction_hidden 512 1024 --dropout 0.1 \
--pretrain_epoches 10000 \
--lr 0.001 \
--batchsize 128 --epoches 10000 --patience 4 --print_every 16 &

