import argparse
import os, sys
import numpy as np
from src.weakSupervised.weakSupervised import pretrain
from src.weakSupervised.data_utils import load_fn
from src.model_subclass import GraphEmbedding, ProtSeqEmbedding, BiInteraction
from src.graphLayer import WeaveGather
from src.data_utils import DataSet, PROTCHARSIZE
from src.utils import make_config_str, PROJPATH
from src.featurizer import WeaveFeaturizer
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras import Model
from itertools import chain
import tensorflow as tf
from src.emetrics import cindex_score
from src.utils import log
from pprint import pprint
from functools import partial
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type= str, default= "kiba", help = "dataset to use in training")
parser.add_argument("--no_pretrain", action= 'store_false', dest= 'pretrain', default= False, help= "whether to use pretrain graph convolution layer")
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
parser.add_argument("--pretrain_epoches", type= int, default= 1, help= "Epoches in pretraining.")
parser.add_argument("--patience", type= int, default= 1, help= "patience epoch to wait during early stopping.")
parser.add_argument("--print_every", type= int, default= 32, help= "print intervals during loop dataset.")

args = parser.parse_args()


pprint(vars(args))
prefix = "dataset~%s/pretrain~%s-lr~%s-batchsize~%s-atom_hidden~%s-pair_hidden~%s-graph_dim~%s-num_filters~%s-biInt_hidden~%s-dropout~%s-epoches~%s/" \
         % (args.dataset, args.pretrain, args.lr, args.batchsize, '_'.join([str(d) for d in args.atom_hidden]), '_'.join([str(d) for d in args.pair_hidden]),
            args.graph_features, '_'.join([str(d) for d in args.num_filters]),
            '_'.join([str(d) for d in args.biInteraction_hidden]) , args.dropout, args.epoches)
# configStr = "test"
chkpt_dir = os.path.join(PROJPATH, "checkpoint/", prefix)
log_dir = os.path.join(PROJPATH, "log/", prefix)
if not os.path.exists(chkpt_dir):
    os.makedirs(chkpt_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_f = open(os.path.join(log_dir, 'log'), 'w')
printf = partial(log, f = log_f)

filepath = os.path.join(PROJPATH, "data/%s/" %(args.dataset, ))
weave_featurizer = WeaveFeaturizer()

PROTSEQLENGTH= 1000
dataset = DataSet(fpath=filepath,
                  seqlen= PROTSEQLENGTH,
                  featurizer=weave_featurizer,
                  is_log=False)
fold5_train, test_inds = dataset.load_5fold_split()
train_inds = list(chain(*fold5_train[:3]))
val_inds = fold5_train[4]
test_inds = test_inds

atom_dim = 75
pair_dim = 14
props_dim = 100

optimizer = tf.train.AdamOptimizer(learning_rate= args.lr)

atom_features = Input(shape=(atom_dim,))
pair_features = Input(shape=(pair_dim,))
pair_split = Input(shape=(), dtype=tf.int32)
atom_split = Input(shape=(), dtype=tf.int32)
atom_to_pair = Input(shape=(2,), dtype=tf.int32)
atoms_input = [atom_features, pair_features, pair_split, atom_split, atom_to_pair]

protSeq = Input(shape=(PROTSEQLENGTH,))

atom_embedding = GraphEmbedding(atom_features= atom_dim,
                                pair_features= pair_dim,
                                atom_hidden_list= args.atom_hidden,
                                pair_hidden_list= args.pair_hidden,
                                graph_feat= args.graph_features,
                                num_mols= args.batchsize,
                                name= 'graph_embedding'
                                )(atoms_input)
mol_embedding = WeaveGather(args.batchsize, atom_dim= args.graph_features, name='atom_gather')([atom_embedding, atom_split])
mol_property = Dense(props_dim, name= 'mol_property')(mol_embedding)
protSeq_embedding = ProtSeqEmbedding(num_filters_list= args.num_filters,
                                           filter_length_list= args.filters_length,
                                           prot_char_size= PROTCHARSIZE,
                                           max_seq_length= PROTSEQLENGTH,
                                           name = 'protein_embedding'
                                           )(protSeq)
affinity = BiInteraction(hidden_list= args.biInteraction_hidden,
                                    dropout= args.dropout,
                                    activation= None,
                                    name= 'biInteraction')([atom_embedding, protSeq_embedding, atom_split])
DrugPropertyModel = Model(inputs= atoms_input, outputs= mol_property, name= 'drugPropertyModel')
DrugPropertyModel.summary() #to test:
DTAModel= Model(inputs = [atoms_input, protSeq],
                outputs= affinity,
                name= "DTAmodel")

tf.enable_eager_execution()
print("pretrain...")
if args.pretrain:
    pretrain(DrugPropertyModel,
             dataset= "kiba_origin",
             dir_prefix= prefix,
             epoches= args.pretrain_epoches,
             batchsize= 64,
             lr = 1E-4,
             patience= 8,
             )
    print("pretrain conclude.")

def loop_dataset(indices, optimizer = None):
    mean_loss = 0
    mean_ci = 0
    count = len(indices) // args.batchsize
    for it, (batch_mol, batch_protSeq, labels) in enumerate(
            dataset.iter_batch(args.batchsize, indices, shuffle=True, )):
        # print(it)

        with tf.GradientTape() as tape:
            logit = DTAModel([batch_mol, batch_protSeq])
            # print(logit.numpy(), labels)
            loss_tensor = tf.losses.mean_squared_error(labels, logit)
            ci_tensor = cindex_score(labels, logit)

        loss_value = loss_tensor.numpy().mean()
        ci_value = ci_tensor.numpy()

        mean_loss = (it * mean_loss + loss_value) / (it + 1)
        mean_ci = (it * mean_ci + ci_value) / (it + 1)
        if optimizer:
            variables = DTAModel.variables
            grads = tape.gradient(loss_tensor, variables)
            optimizer.apply_gradients(zip(grads, variables), global_step=tf.train.get_or_create_global_step())

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
        DTAModel.save_weights(os.path.join(chkpt_dir, "DTA"), )
        wait = 0
    else:
        wait += 1

    if wait > args.patience:
        break

DTAModel.load_weights(os.path.join(chkpt_dir, "DTA"))

printf("start testing...")
test_loss, test_ci = loop_dataset(test_inds, optimizer= None)
printf("test loss: %.4f ci: %.4f \n" %(test_loss, test_ci))
