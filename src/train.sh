#!/usr/bin/env bash
python train.py --dataset kiba \
--atom_hidden 32 32 --pair_hidden 16 8 --graph_features 32 \
--num_filters 32 64 96 --filters_length 4 8 12 \
--biInteraction_hidden 1024 1024 512 --dropout 0.1 \
--batchsize 256 --epoches 10000 --patience 4 --print_every 16 --lr 0.001
