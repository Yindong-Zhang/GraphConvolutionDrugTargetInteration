#!/usr/bin/env bash
python -m src.train --dataset kiba \
--atom_hidden 64 64 32 --pair_hidden 32 32 16 --graph_features 256 \
--num_filters 32 64 96 --filters_length 4 12 8 \
--biInteraction_hidden 1024 1024 512 --dropout 0.1 \
--pretrain_epoches 10000 \
--batchsize 32 --epoches 10000 --patience 4 --print_every 16 \
#--no_pretrain

