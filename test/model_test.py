from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras import Model
import tensorflow as tf
from src.model import build_graph_embedding_model, build_protSeq_embedding_model
from src.featurizer import WeaveFeaturizer
from src.data_utils import DataSet
import numpy as np
from itertools import chain
tf.enable_eager_execution()

def graph_embedding_test():
    filepath = '../data/kiba/'
    weave_featurizer = WeaveFeaturizer()
    dataset = DataSet(fpath=filepath,
                      setting_no=1,  ##
                      seqlen=1000,
                      featurizer=weave_featurizer,
                      is_log= False)
    fold5_train, test_inds = dataset.load_5fold_split()
    train_inds = list(chain(*fold5_train[:3]))
    test_inds = np.array(test_inds)

    batchsize = 32
    atom_dim = 75
    pair_dim = 14
    atom_features = Input(shape=(atom_dim,))
    pair_features = Input(shape=(pair_dim,))
    pair_split = Input(shape=(), dtype=tf.int32)
    atom_split = Input(shape=(), dtype=tf.int32)
    atom_to_pair = Input(shape=(2,), dtype=tf.int32)
    graph_embed_model = build_graph_embedding_model(atom_dim, pair_dim, [16, 16, 8], [8, 4, 4], 8, batchsize)
    graph_embed = graph_embed_model([atom_features, pair_features, pair_split, atom_split, atom_to_pair])
    graph_label = Dense(1, activation= 'sigmoid')(graph_embed)
    model = Model(inputs = [atom_features, pair_features, pair_split, atom_split, atom_to_pair],
                  outputs = graph_label)
    # The compile step specifies the training configuration.
    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss='mse',
                  )
    model.summary()

    optimizer = tf.train.AdamOptimizer()
    loss_values = []
    for epoch in range(2):
        for it, (batch_mol, batch_protSeq, labels) in enumerate(dataset.iter_batch(batchsize, train_inds, shuffle=True, )):
            print(it)

            with tf.GradientTape() as tape:

                pred = model(batch_mol)
                # print(pred.numpy())
                loss_value = tf.losses.mean_squared_error(labels, pred)

            loss_values.append(loss_value.numpy())
            print(loss_value.numpy())
            grads = tape.gradient(loss_value, model.variables)
            optimizer.apply_gradients(zip(grads, model.variables), global_step= tf.train.get_or_create_global_step())

def protSeqEmbedding_test():
    filepath = '../data/kiba/'
    weave_featurizer = WeaveFeaturizer()
    dataset = DataSet(fpath=filepath,  ### BUNU ARGS DA GUNCELLE
                      setting_no=1,  ##BUNU ARGS A EKLE
                      seqlen=1000,
                      featurizer=weave_featurizer,
                      is_log= False)
    fold5_train, test_inds = dataset.load_5fold_split()
    train_inds = list(chain(*fold5_train[:3]))
    test_inds = np.array(test_inds)

    batchsize = 32
    protseqembed = build_protSeq_embedding_model([8, 4, 4], [8, 16, 8], 25, 1000)
    pass

if __name__ == "__main__":
    graph_embedding_test()
