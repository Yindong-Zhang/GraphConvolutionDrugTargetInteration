from src.model_subclass import GraphEmbedding, ProtSeqEmbedding
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from deepchem.feat import WeaveFeaturizer
from src.data_utils import DataSet
from itertools import chain
import numpy as np


tf.enable_eager_execution()

def protSeqEmbedding_test():
    filepath = '../data/kiba/'
    weave_featurizer = WeaveFeaturizer()
    dataset = DataSet(fpath=filepath,  ### BUNU ARGS DA GUNCELLE
                      problem_type=1,  ##BUNU ARGS A EKLE
                      seqlen=1000,
                      featurizer=weave_featurizer,
                      is_log= False)
    fold5_train, test_inds = dataset.load_5fold_split()
    train_inds = list(chain(*fold5_train[:3]))
    test_inds = np.array(test_inds)

    batchsize = 32
    model = ProtSeqEmbedding([8, 4, 4], [8, 16, 8], 25, 1000)


    optimizer = tf.train.AdamOptimizer()
    loss_values = []
    for epoch in range(2):
        for it, (batch_mol, batch_protSeq, labels) in enumerate(dataset.iter_batch(batchsize, train_inds, shuffle=True, )):
            print(it)

            with tf.GradientTape() as tape:

                protEmbed = model(batch_protSeq)
                dense = Dense(1)
                logit = dense(protEmbed)
                # print(pred.numpy())
                loss_tensor = tf.losses.mean_squared_error(labels, logit)

            loss_value = loss_tensor.numpy()
            loss_values.append(loss_value)
            print(loss_value)
            grads = tape.gradient(loss_tensor, model.variables + dense.variables)
            optimizer.apply_gradients(zip(grads, model.variables), global_step= tf.train.get_or_create_global_step())


def graphEmbedding_test():
    filepath = '../data/kiba/'
    weave_featurizer = WeaveFeaturizer()
    dataset = DataSet(fpath=filepath,  ### BUNU ARGS DA GUNCELLE
                      problem_type=1,  ##BUNU ARGS A EKLE
                      seqlen=1000,
                      featurizer=weave_featurizer,
                      is_log=False)
    fold5_train, test_inds = dataset.load_5fold_split()
    train_inds = list(chain(*fold5_train[:3]))
    test_inds = np.array(test_inds)

    batchsize = 2
    atom_dim = 75
    pair_dim = 14
    model = GraphEmbedding(atom_features= atom_dim,
                           pair_features= pair_dim,
                           atom_hidden_list= [16, 16, 16],
                           pair_hidden_list= [4, 4, 2],
                           graph_feat= 8,
                           num_mols= batchsize)
    optimizer = tf.train.AdamOptimizer(learning_rate= 0.1)
    loss_values = []
    for epoch in range(2):
        for it, (batch_mol, batch_protSeq, labels) in enumerate(
                dataset.iter_batch(batchsize, train_inds, shuffle=True, )):
            print(it)

            with tf.GradientTape() as tape:
                protEmbed = model(batch_mol)
                dense = Dense(1, activation= None, use_bias= False)
                logit = dense(protEmbed)
                print(logit.numpy(), labels)
                loss_tensor = tf.losses.mean_squared_error(labels, logit)

            loss_value = loss_tensor.numpy()
            loss_values.append(loss_value)
            print(loss_value)
            vars = model.variables + dense.variables
            grads = tape.gradient(loss_tensor, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step=tf.train.get_or_create_global_step())

if __name__ == "__main__":
    # protSeqEmbedding_test()
    graphEmbedding_test()