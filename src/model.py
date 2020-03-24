import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, BatchNormalization, Embedding, Conv1D, GlobalMaxPooling1D, Input, Concatenate, Dropout
from src.graphLayer import WeaveLayer, WeaveGather

def build_graph_embedding_model(atom_dim, pair_dim, atom_hidden_list, pair_hidden_list, graph_features, num_mols, name= None):
    atom_features = Input(shape=(atom_dim,))
    pair_features = Input(shape=(pair_dim,))
    pair_split = Input(shape=(), dtype=tf.int32)
    atom_split = Input(shape=(), dtype=tf.int32)
    atom_to_pair = Input(shape=(2,), dtype=tf.int32)
    num_weaveLayers = len(atom_hidden_list)

    atom_hidden, pair_hidden = WeaveLayer(atom_dim, pair_dim, atom_hidden_list[0], pair_hidden_list[0])([atom_features, pair_features, pair_split, atom_to_pair])
    for i in range(1, num_weaveLayers):
        atom_hidden, pair_hidden= WeaveLayer(atom_hidden_list[i - 1], pair_hidden_list[i - 1], atom_hidden_list[i], pair_hidden_list[i])([atom_hidden, pair_hidden, pair_split, atom_to_pair])

    atom_hidden = Dense(graph_features, activation='tanh')(atom_hidden)
    atom_hidden = BatchNormalization()(atom_hidden)
    graph_embed = WeaveGather(num_mols, graph_features, gaussian_expand=True)([atom_hidden, atom_split])

    model = tf.keras.Model(
        inputs=[
            atom_features, pair_features, pair_split, atom_split, atom_to_pair
        ],
        outputs= graph_embed,
        name= name,
    )
    return model


def buildprotSeq_embedding_model(num_filters_list, filter_length_list, prot_seq_dim, embed_size, max_seq_length, name= None):
    assert len(num_filters_list) == len(filter_length_list), "incompatible hyper parameter."
    num_conv_layers = len(num_filters_list)
    protSeq = Input(shape= (max_seq_length, ))
    seq_embed = Embedding(input_dim=prot_seq_dim + 1, output_dim=embed_size, input_length=max_seq_length)(protSeq)
    for i in range(num_conv_layers):
        seq_embed = Conv1D(filters=num_filters_list[i], kernel_size= filter_length_list[i], activation='relu', padding='valid', strides=1)(seq_embed)
    seq_embed = GlobalMaxPooling1D()(seq_embed)
    model = Model(inputs = protSeq,
                  outputs = seq_embed,
                  name= name)
    return model

def BiInteraction(graph_dim, seq_dim, name= None):
    graph_embed = Input(shape= (graph_dim, ))
    protSeq_embed = Input(shape= (seq_dim, ))
    feat_concat = Concatenate()([graph_embed, protSeq_embed])
    int_embed = Dense(1024, activation='relu')(feat_concat)
    int_embed = Dropout(0.1)(int_embed)
    int_embed = Dense(1024, activation='relu')(int_embed)
    aff_logit = Dense(1, activation= None)(int_embed)
    model = Model(inputs = [graph_embed, protSeq_embed],
                  outputs = aff_logit,
                  name= name)
    return model



if __name__ == "__main__":
    model = build_protSeq_embedding_model([4, 8, 8], [5, 7, 9], 255, 1000)
    model = build_graph_embedding_model(16, 8, [8, 8], [4, 4], 16, 8)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )
    model.summary()
