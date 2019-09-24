import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, BatchNormalization, Embedding, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout, Layer
from tensorflow.python.keras import initializers
from src.graphLayer import WeaveLayer, WeaveGather, MolecularConvolutionLayer

""" model subclass is more suitable for tensorflow 2.0
"""
class GraphEmbedding(Model):
    def __init__(self, atom_features, pair_features, atom_hidden_list, pair_hidden_list, graph_feat, num_mols, *args, **kwargs):
        super(GraphEmbedding, self).__init__(*args, **kwargs)
        self.atom_hidden_list = atom_hidden_list
        self.pair_hidden_list = pair_hidden_list
        self.graph_features = graph_feat
        self.num_mols = num_mols
        assert len(atom_hidden_list) == len(pair_hidden_list), "length of atom hidden list should equal to length of pair hidden list."
        self.num_GCNLayers = len(atom_hidden_list)

        self.atom_features = atom_features
        self.pair_features = pair_features

        self.GCLayer_list = []

        # TODO:
        atom_dim_sum = 75
        pair_dim_sum = 14
        self.atom_hidden_input_list = []
        self.pair_hidden_input_list = []
        for i in range(self.num_GCNLayers):
            atom_dim_sum = atom_dim_sum + self.atom_hidden_list[i]
            pair_dim_sum = pair_dim_sum + self.pair_hidden_list[i]
            self.atom_hidden_input_list.append(atom_dim_sum)
            self.pair_hidden_input_list.append(pair_dim_sum)
        # self.GCLayer_list.append(MolecularConvolutionLayer(self.atom_features, self.pair_features, self.atom_hidden_list[0], self.pair_hidden_list[0], self.atom_hidden_list[0]))
        self.GCLayer_list.append(WeaveLayer(self.atom_features, self.pair_features, self.atom_hidden_list[0], self.pair_hidden_list[0], activation= 'tanh'))
        for i in range(1, self.num_GCNLayers):
            # self.GCLayer_list.append(
            #     MolecularConvolutionLayer(self.atom_hidden_list[i - 1], self.pair_hidden_list[i - 1], self.atom_hidden_list[i], self.pair_hidden_list[i], self.atom_hidden_list[i]))
            self.GCLayer_list.append(
                WeaveLayer(self.atom_hidden_input_list[i - 1], self.pair_hidden_input_list[i - 1], self.atom_hidden_list[i], self.pair_hidden_list[i], activation= 'tanh'))

        self.dense = Dense(self.graph_features, activation='tanh')
        self.batchnorm = BatchNormalization()

    def call(self, inputs):
        atom_features, pair_features, pair_split, atom_split, atom_to_pair = inputs
        atom_hidden_list = []
        pair_hidden_list = []
        atom_hidden, pair_hidden = atom_features, pair_features
        atom_hidden_list.append(atom_hidden)
        pair_hidden_list.append(pair_hidden)
        for i in range(self.num_GCNLayers):
            atom_hidden = tf.concat(atom_hidden_list, axis= -1)
            pair_hidden = tf.concat(pair_hidden_list, axis= -1)
            atom_hidden, pair_hidden = self.GCLayer_list[i]([atom_hidden, pair_hidden, pair_split, atom_to_pair])
            atom_hidden_list.append(atom_hidden)
            pair_hidden_list.append(pair_hidden)
        atom_hidden_out = tf.concat(atom_hidden_list, axis= -1)
        atom_hidden = self.dense(atom_hidden_out)
        atom_hidden = self.batchnorm(atom_hidden)
        return atom_hidden

    def compute_output_shape(self, input_shape):
        atom_shape, _, _, _, _ = input_shape
        num_atoms, atom_dim = atom_shape
        return tf.TensorShape([num_atoms, self.graph_features])

class ProtSeqEmbedding(Model):
    def __init__(self, num_filters_list, filter_length_list, prot_char_size, max_seq_length, **kwargs):
        super(ProtSeqEmbedding, self).__init__(**kwargs)
        self.prot_seq_length = prot_char_size
        self.max_seq_length = max_seq_length
        assert len(num_filters_list) == len(filter_length_list), "Incompatibe hyperparameters."
        self.num_conv_layers = len(num_filters_list)
        self.num_filters_list = num_filters_list
        self.filter_length_list = filter_length_list
        self.embed =  Embedding(input_dim=prot_char_size + 1, output_dim=128, input_length= max_seq_length)
        self.conv_layer_list = []
        for i in range(self.num_conv_layers):
            self.conv_layer_list.append(Conv1D(filters= num_filters_list[i],
                                                     kernel_size= filter_length_list[i],
                                                     activation='relu',
                                                     padding='same',
                                                     strides=1))

    def call(self, inputs):
        seq_embed = self.embed(inputs)
        for layer in self.conv_layer_list:
            seq_embed = layer(seq_embed)

        return seq_embed

    def compute_output_shape(self, input_shape):
        batchsize, protSeqLen = input_shape
        embed_dim = self.num_filters_list[-1]

        return tf.TensorShape([batchsize, protSeqLen, embed_dim])

class BiInteraction(Layer):
    def __init__(self, hidden_list, dropout, activation = "relu", **kwargs):
        super(BiInteraction, self).__init__(**kwargs)
        self.num_dense_layers = len(hidden_list)
        self.activation = activation
        self.dropout= dropout
        self.dense_layer_list = []
        for i in range(self.num_dense_layers):
            self.dense_layer_list.append(Dense(hidden_list[i], activation= activation))
        self.out_layer = Dense(1)

    def build(self, input_shape):
        atom_hidden_shape, prot_hidden_shape, atom_splits_shape = input_shape
        atom_dim = atom_hidden_shape[-1]
        prot_dim = prot_hidden_shape[-1]
        self.W= self.add_weight('attention_weight', shape= (atom_dim, prot_dim), initializer= initializers.get(self.activation))

    def call(self, inputs, training= None):
        atom_embed, protSeq_embed, atom_splits = inputs

        protSeq_embed_T = tf.transpose(protSeq_embed, (0, 2, 1))
        protSeq_embed_gather = tf.gather(protSeq_embed_T, atom_splits, axis= 0)
        W = tf.einsum('ij, jk, ikl->il', atom_embed, self.W, protSeq_embed_gather)

        Wc = tf.exp(tf.reduce_max(W, axis= -1 ,keepdims= True))
        Sc = tf.gather(tf.segment_sum(Wc, atom_splits), atom_splits, axis= 0)
        aa = Wc / Sc

        atom_embed = tf.segment_sum(aa * atom_embed, atom_splits)

        Wp = tf.segment_max(W, atom_splits)
        ap = tf.nn.softmax(Wp, axis= -1)

        prot_embed = tf.einsum('ij, ijk->ik', ap, protSeq_embed)

        concat_embed = tf.concat([atom_embed, prot_embed], axis = -1)
        for layer in self.dense_layer_list:
            concat_embed = layer(concat_embed)
            if training:
                concat_embed = Dropout(self.dropout)(concat_embed)
        return self.out_layer(concat_embed)

    def compute_output_shape(self, input_shape):
        (num_atoms, atom_dim), (batchsize, prot_dim), (num_atoms, ) = input_shape
        return tf.TensorShape((batchsize, 1))


if __name__ == "__main__":
    tf.enable_eager_execution()

    graph_embed_model = GraphEmbedding([8, 8], [4, 4], 16, 8)
    # graph_embed_model.build([tf.TensorShape((None, 16, )),
    #                          tf.TensorShape((None, 20, )),
    #                          tf.TensorShape((None, )),
    #                          tf.TensorShape((None, )),
    #                          tf.TensorShape((None, 2,))]
    #                         )
    # graph_embed_model.compile(optimizer= 'adam',
    #                           loss='categorical_crossentropy',
    #                           metrics=['accuracy']
    #                           )
    graph_embed_model.summary()
