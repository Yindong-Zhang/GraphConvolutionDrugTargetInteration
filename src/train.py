import argparse
import os, sys
from shutil import copy
import numpy as np
from src.weakSupervised.weakSupervised import pretrain
from src.weakSupervised.data_utils import load_fn
from src.model_subclass import GraphEmbedding, ProtSeqEmbedding, BiInteraction, ConcatMlp, EmbeddingLayer, Concatenate
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
parser.add_argument("--dataset", type= str, default= "davis", help = "dataset to use in training")
parser.add_argument("--pretrain", action= 'store_true', dest= 'pretrain', default= False, help= "whether to use pretrain graph convolution layer")
parser.add_argument("--lr", type= float, default= 0.001, help= "learning rate in optimizer")
parser.add_argument("--batchsize", type= int, default= 64, help = "batchsize during training.")
parser.add_argument("--atom_hidden", type= int, nargs= "+", default= [256, 512], help= "atom hidden dimension list in graph embedding model.")
parser.add_argument("--pair_hidden", type= int, nargs= "+", default= [256, 512], help = "pair hidden dimension list in graph embedding model.")
parser.add_argument("--graph_features", type= int, default= 512, help= "graph features dimension")
parser.add_argument("--num_filters", type= int, nargs= "+", default= [32, 64, 128], help = "numbers of 1D convolution filters protein seq embedding model.")
parser.add_argument("--filters_length", type= int, nargs= "+", default= [4, 8, 12], help = "filter length list of 1D conv filters in protSeq embedding.")
parser.add_argument("--mol_embed_size", type= int, default= 32, help = 'molecular atom and bond feature embed size')
parser.add_argument("--biInteraction_hidden", type= int, nargs= "+", default= [512, 1024], help = "hidden dimension list in BiInteraction model.")
parser.add_argument("--dropout", type= float, default= 0.1, help= "dropout rate in biInteraction model.")
parser.add_argument("--epoches", type= int, default= 2, help= "epoches during training..")
parser.add_argument("--pretrain_epoches", type= int, default= 1, help= "Epoches in pretraining.")
parser.add_argument("--patience", type= int, default= 1, help= "patience epoch to wait during early stopping.")
parser.add_argument("--print_every", type= int, default= 1, help= "print intervals during loop dataset.")

args = parser.parse_args()
tf.enable_eager_execution()

pprint(vars(args))
prefix = "dataset~%s/pretrain~%s-lr~%s-batchsize~%s-atom_hidden~%s-pair_hidden~%s-graph_dim~%s-num_filters~%s-biInt_hidden~%s-dropout~%s-epoches~%s-weave-cv5/" \
         % (args.dataset, args.pretrain, args.lr, args.batchsize, '_'.join([str(d) for d in args.atom_hidden]), '_'.join([str(d) for d in args.pair_hidden]),
            args.graph_features, '_'.join([str(d) for d in args.num_filters]),
            '_'.join([str(d) for d in args.biInteraction_hidden]) , args.dropout, args.epoches)
print(prefix)
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
mol_featurizer = WeaveFeaturizer()

PROTSEQLENGTH= 1000
dataset = DataSet(fpath=filepath,
                  seqlen= PROTSEQLENGTH,
                  featurizer=mol_featurizer,
                  is_log= args.dataset == 'davis',)
fold5_train, test_inds = dataset.load_5fold_split()
test_inds = test_inds

atom_dim = len(mol_featurizer.atom_cat_dim) * args.mol_embed_size
pair_dim = len(mol_featurizer.bond_cat_dim) * args.mol_embed_size
props_dim = 100



atom_features = [Input(shape=()) for _ in range(len(mol_featurizer.atom_cat_dim))]
pair_features = [Input(shape=()) for _ in range(len(mol_featurizer.bond_cat_dim))]
pair_split = Input(shape=(), dtype=tf.int32)
atom_split = Input(shape=(), dtype=tf.int32)
atom_to_pair = Input(shape=(2,), dtype=tf.int32)
num_atoms = Input(shape=(), dtype= tf.int32, batch_size= 1)
mol_input = [atom_features, pair_features, pair_split, atom_split, atom_to_pair, num_atoms]

protSeq = Input(shape=(PROTSEQLENGTH,))

atom_feat = EmbeddingLayer(mol_featurizer.atom_cat_dim, args.mol_embed_size)(atom_features)

pair_feat = EmbeddingLayer(mol_featurizer.bond_cat_dim, args.mol_embed_size)(pair_features)
mol_feat = [atom_feat, pair_feat, pair_split, atom_split, atom_to_pair, num_atoms]

atom_embedding = GraphEmbedding(atom_features= atom_dim,
                                pair_features= pair_dim,
                                atom_hidden_list= args.atom_hidden,
                                pair_hidden_list= args.pair_hidden,
                                graph_feat= args.graph_features,
                                num_mols= args.batchsize,
                                name= 'graph_embedding'
                                )(mol_feat)
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
                                    activation= 'tanh',
                                    name= 'biInteraction')([atom_embedding, protSeq_embedding, atom_split])
# affinity = ConcatMlp(hidden_list= args.biInteraction_hidden, activation= 'tanh')([atom_embedding, protSeq_embedding, atom_split])
# DrugPropertyModel = Model(inputs= mol_input, outputs= mol_property, name= 'drugPropertyModel')
DTAModel= Model(inputs = [mol_input, protSeq],
                outputs= affinity,
                name= "DTAmodel")
init_weight_subdir = chkpt_dir + '/initial/'
if not os.path.exists(init_weight_subdir):
    os.makedirs(init_weight_subdir)

# DTAModel.save_weights(init_weight_subdir)
init_weight = DTAModel.get_weights()
if args.pretrain:
    print("pretrain...")
    DrugPropertyModel.summary()
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
    isTraining = optimizer is not None
    for it, (batch_mol, batch_protSeq, labels) in enumerate(
            dataset.iter_batch(args.batchsize, indices, shuffle=False)):
        # print(it)
        # print(batch_mol[0])
        with tf.GradientTape() as tape:
            logit = DTAModel([batch_mol, batch_protSeq], training= isTraining)
            # print(logit.numpy())
            loss_tensor = tf.losses.mean_squared_error(labels, logit)
            ci_tensor = cindex_score(labels, logit)

        loss_value = loss_tensor.numpy().mean()
        ci_value = ci_tensor.numpy()

        mean_loss = (it * mean_loss + loss_value) / (it + 1)
        mean_ci = (it * mean_ci + ci_value) / (it + 1)
        if optimizer:
            variables = DTAModel.variables
            grads = tape.gradient(loss_tensor, variables)
            optimizer.apply_gradients(zip(grads, variables), global_step= global_step) # problem here?
            # print(tf.train.get_or_create_global_step().numpy().item())

        if it % args.print_every == 0:
            printf("%s / %s: mean_loss: %.4f ci: %.4f. " %(it, count, mean_loss, mean_ci))

        # if it > 10:
        #     break

    return mean_loss, mean_ci


global_step = tf.train.create_global_step()

DTAModel.summary()
best_ci = float('-inf')
best_loss = float('inf')
best_it = -1
for it in range(5):
    # if it != 1:
    #     continue
    val_inds = fold5_train[it]
    train_inds = []
    for ind in range(5):
        if ind != it:
            train_inds.extend(fold5_train[ind])
    print(max(train_inds), min(train_inds))
    printf("Start cross validation %d..." %(it, ))
    chkpt_subdir = chkpt_dir + '/cv~%d/' %(it, )
    if not os.path.exists(chkpt_subdir):
        os.makedirs(chkpt_subdir)
    DTAModel.set_weights(init_weight)

    global_step.assign(0)
    optimizer = tf.train.AdamOptimizer(learning_rate=args.lr) # careful here for optimizer record moving mean velocity and mass

    best_metric = float("inf")
    wait = 0
    for epoch in range(args.epoches):
        printf("training epoch %s..." %(epoch, ))
        train_loss, train_ci = loop_dataset(train_inds, optimizer = optimizer)
        printf("train epoch %d loss %.4f ci %.4f \n" %(epoch, train_loss, train_ci))

        printf("validating epoch %s..." %(epoch, ))
        val_loss, val_ci = loop_dataset(val_inds, optimizer= None)
        printf("validating epoch %d loss %.4f ci %.4f \n" %(epoch, val_loss, val_ci))
        if val_loss <= best_metric:
            best_metric = val_loss
            DTAModel.save_weights(os.path.join(chkpt_subdir, "DTA"), )
            wait = 0
        else:
            wait += 1

        if wait > args.patience:
            break

    DTAModel.load_weights(os.path.join(chkpt_subdir, "DTA"))

    printf("start testing...")
    test_loss, test_ci = loop_dataset(test_inds, optimizer= None)
    if test_ci > best_ci:
        best_ci = test_ci
        best_loss= test_loss
        best_it = it
    printf("CV %d test loss: %.4f ci: %.4f \n" %(it, test_loss, test_ci))

printf("Best iteration in fold-5 CV: %d, Best loss: %.4f, Best CI: %.4f." %(best_it, best_loss, best_ci))
