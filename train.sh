#!/usr/bin/env bash
echo '' > nohup.out
nohup \
python -m src.train --dataset davis \
--atom_hidden 128 256 256   --pair_hidden 128 256 256  --graph_features 256 \
--num_filters 64 96 128 --filters_length 4 8 12 \
--biInteraction_hidden 1024 256  --dropout 0.2 \
--pretrain_epoches 10 \
--weight_decay 5E-6 \
--lr 0.001 \
--batchsize 64 --epoches 1000 --patience 4 --print_every 16 \
&

