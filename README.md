# About graphConvolutionDTA: graph convolution drug-target binding affinity prediction
This repository contains drug-target binding affinity prediction using graph convolution in molecular side, similar to ["Compound-protein Interaction Prediction with End-to-end Learning of Neural Networks for Graphs and Sequences (Bioinformatics, 2018)"](https://github.com/masashitsubaki/CPI_prediction)


The approach used in this work model protein sequences  using convolutional neural networks (CNNs) and compound molecular graph using  graph convolution network(WeaveLayer for example) to predict the binding affinity value of drug-target pairs.


I use dataset from [deepDTA](http://arxiv.org/abs/1801.10193), which use convolution network on both protein sequence and compound SMILES sequence reached best performance of 0.863 (CI) on KIBA Dataset. This method using molecular graph convolution(WeaveModule for example) achieve at best 0.874 on Davis Dataset and 0.866 on KIBA Dataset.


* Different depth and size of graph convolution layer are searched
* atom and bond categorical features are encoded by embedding layer and concated as initial input to GCN layers.
* A pretraining experiment on graph convolution side using calculated molecular property, see a little improvement about 0.002, idea from [http://arxiv.org/abs/1712.02734] .pretraing on [KIBA origin dataset](http://arxiv.org/abs/1801.10193).
* a biInteraction layer between compound atom vectors following GCN and protein sequence vector following CNN, similar to [http://arxiv.org/abs/1806.07537], but it's hard to train and see barely any improvement.
## Requirements

You'll need to install following in order to run the codes.

*  [Python 3.6]
*  [Tensorflow 1.14](https://www.tensorflow.org/install/)
*  numpy
*  matplotlib
