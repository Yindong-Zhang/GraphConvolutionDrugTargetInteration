#!/usr/bin/env bash
echo '' > nohup.out
 nohup \
python -m src.train --dataset davis \
--atom_hidden 128 256 512   --pair_hidden 128 256 512  --graph_features 512 \
--num_filters 64 128 256 --filters_length 4 8 12 \
--biInteraction_hidden 512 1024 --dropout 0.2 \
--pretrain_epoches 10 \
--weight_decay 5E-6 \
--lr 0.001 \
--batchsize 64 --epoches 1000 --patience 4 --print_every 16 \
 &

