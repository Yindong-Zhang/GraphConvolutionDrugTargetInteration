import argparse
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import numpy as np
from src.model_subclass import GraphEmbedding, ProtSeqEmbedding, BiInteraction
from src.data_utils import DataSet, PROTCHARSIZE
from src.utils import make_config_str
from src.featurizer import WeaveFeaturizer
from itertools import chain
import tensorflow as tf
from src.emetrics import cindex_score
from src.utils import log
from pprint import pprint
from functools import partial
tf.enable_eager_execution()
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type= str, default= "kiba", help = "dataset to use in training")
parser.add_argument("--lr", type= float, default= 0.001, help= "learning rate in optimizer")
parser.add_argument("--batchsize", type= int, default= 32, help = "batchsize during training.")
parser.add_argument("--atom_hidden", type= int, nargs= "+", default= [32, 16], help= "atom hidden dimension list in graph embedding model.")
parser.add_argument("--pair_hidden", type= int, nargs= "+", default= [16, 8], help = "pair hidden dimension list in graph embedding model.")
parser.add_argument("--graph_features", type= int, default= 32, help= "graph features dimension")
parser.add_argument("--num_filters", type= int, nargs= "+", default= [32, 16], help = "numbers of 1D convolution filters protein seq embedding model.")
parser.add_argument("--filters_length", type= int, nargs= "+", default= [16, 32], help = "filter length list of 1D conv filters in protSeq embedding.")
parser.add_argument("--biInteraction_hidden", type= int, nargs= "+", default= [128, 16], help = "hidden dimension list in BiInteraction model.")
parser.add_argument("--dropout", type= float, default= 0.1, help= "dropout rate in biInteraction model.")
parser.add_argument("--epoches", type= int, default= 2, help= "epoches during training..")
parser.add_argument("--patience", type= int, default= 1, help= "patience epoch to wait during early stopping.")
parser.add_argument("--print_every", type= int, default= 32, help= "print intervals during loop dataset.")
args = parser.parse_args()


pprint(vars(args))
configStr = make_config_str(args)
# configStr = "test"
chkpt_dir = os.path.join("../checkpoint", configStr)
if not os.path.exists(chkpt_dir):
    os.makedirs(chkpt_dir)

log_f = open("../log/%s.log" %(configStr, ), 'w')
printf = partial(log, f = log_f)

filepath = "../data/%s" %(args.dataset, )
weave_featurizer = WeaveFeaturizer()

PROTSEQLENGTH= 1000
dataset = DataSet(fpath=filepath,  ### BUNU ARGS DA GUNCELLE
                  seqlen= PROTSEQLENGTH,
                  featurizer=weave_featurizer,
                  is_log=False)
fold5_train, test_inds = dataset.load_5fold_split()
train_inds = list(chain(*fold5_train[:3]))
val_inds = fold5_train[4]
test_inds = test_inds

atom_dim = 75
pair_dim = 14

optimizer = tf.train.AdamOptimizer(learning_rate= args.lr)
graph_embedding_model = GraphEmbedding(atom_features= atom_dim,
                                       pair_features= pair_dim,
                                       atom_hidden_list= args.atom_hidden,
                                       pair_hidden_list= args.pair_hidden,
                                       graph_feat= args.atom_hidden[-1],
                                       num_mols= args.batchsize)
protSeq_embedding_model = ProtSeqEmbedding(num_filters_list= args.num_filters,
                                           filter_length_list= args.filters_length,
                                           prot_char_size= PROTCHARSIZE,
                                           max_seq_length= PROTSEQLENGTH,
                                           )
biInteraction_model = BiInteraction(hidden_list= args.biInteraction_hidden, dropout= args.dropout, activation= None)


def loop_dataset(indices, optimizer = None):
    mean_loss = 0
    mean_ci = 0
    count = len(indices) // args.batchsize
    for it, (batch_mol, batch_protSeq, labels) in enumerate(
            dataset.iter_batch(args.batchsize, indices, shuffle=True, )):
        # print(it)

        with tf.GradientTape() as tape:
            graph_embed = graph_embedding_model(batch_mol)
            prot_embed = protSeq_embedding_model(batch_protSeq)
            logit = biInteraction_model(graph_embed, prot_embed)
            # print(logit.numpy(), labels)
            loss_tensor = tf.losses.mean_squared_error(labels, logit)
            ci_tensor = cindex_score(labels, logit)

        loss_value = loss_tensor.numpy().mean()
        ci_value = ci_tensor.numpy()

        mean_loss = (it * mean_loss + loss_value) / (it + 1)
        mean_ci = (it * mean_ci + ci_value) / (it + 1)
        if optimizer:
            vars = graph_embedding_model.variables + protSeq_embedding_model.variables + biInteraction_model.variables
            grads = tape.gradient(loss_tensor, vars)
            optimizer.apply_gradients(zip(grads, vars), global_step=tf.train.get_or_create_global_step())

        if it % args.print_every == 0:
            printf("%s / %s: mean_loss: %.4f ci: %.4f. " %(it, count, mean_loss, mean_ci))


    return mean_loss, mean_ci

best_metric = float("inf")
wait = 0
for epoch in range(args.epoches):
    printf("training epoch %s..." %(epoch, ))
    train_loss, train_ci = loop_dataset(train_inds, optimizer = optimizer)
    printf("train epoch %.4f loss %.4f ci %.4f \n" %(epoch, train_loss, train_ci))

    printf("validating epoch %s..." %(epoch, ))
    val_loss, val_ci = loop_dataset(val_inds, optimizer= None)
    printf("validating epoch %.4f loss %.4f ci %.4f \n" %(epoch, val_loss, val_ci))
    if val_loss < best_metric:
        best_metric = val_loss
        graph_embedding_model.save_weights(os.path.join(chkpt_dir, "graph_embedding_model"), )
        protSeq_embedding_model.save_weights(os.path.join(chkpt_dir, "protSeq_embedding_model"))
        biInteraction_model.save_weights(os.path.join(chkpt_dir, "biInteraction_model"))
        wait = 0
    else:
        wait += 1

    if wait > args.patience:
        break

graph_embedding_model.load_weights(os.path.join(chkpt_dir, "graph_embedding_model"))
protSeq_embedding_model.load_weights(os.path.join(chkpt_dir, "protSeq_embedding_model"))
biInteraction_model.load_weights(os.path.join(chkpt_dir, "biInteraction_model"))

printf("start testing...")
test_loss, test_ci = loop_dataset(test_inds, optimizer= None)
printf("test loss: %.4f ci: %.4f \n" %(test_loss, test_ci))
