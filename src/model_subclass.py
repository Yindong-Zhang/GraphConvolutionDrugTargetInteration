import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, BatchNormalization, Embedding, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout
from src.graphLayer import WeaveLayer, WeaveGather

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
        self.num_weaveLayers = len(atom_hidden_list)

        self.atom_features = atom_features
        self.pair_features = pair_features

        self.weaveLayer_list = []

        self.weaveLayer_list.append(WeaveLayer(self.atom_features, self.pair_features, self.atom_hidden_list[0], self.pair_hidden_list[0]))
        for i in range(1, self.num_weaveLayers):
            self.weaveLayer_list.append(
                WeaveLayer(self.atom_hidden_list[i - 1], self.pair_hidden_list[i - 1], self.atom_hidden_list[i], self.pair_hidden_list[i]))

        self.dense = Dense(self.graph_features, activation='tanh')
        self.batchnorm = BatchNormalization()
        self.weavegather = WeaveGather(self.num_mols, self.graph_features, gaussian_expand=True)

    def call(self, inputs):
        atom_features, pair_features, pair_split, atom_split, atom_to_pair = inputs
        atom_hidden, pair_hidden = self.weaveLayer_list[0]([atom_features, pair_features, pair_split, atom_to_pair])
        for i in range(1, self.num_weaveLayers):
            atom_hidden, pair_hidden = self.weaveLayer_list[i]([atom_hidden, pair_hidden, pair_split, atom_to_pair])
        atom_hidden = self.dense(atom_hidden)
        atom_hidden = self.batchnorm(atom_hidden)
        graph_feat = self.weavegather([atom_hidden, atom_split])
        return graph_feat

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([self.num_mols, self.graph_features])

class ProtSeqEmbedding(Model):
    def __init__(self, num_filters_list, filter_length_list, prot_char_size, max_seq_length):
        super(ProtSeqEmbedding, self).__init__()
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
        self.aggregate_layer = GlobalMaxPooling1D()

    def call(self, inputs):
        seq_embed = self.embed(inputs)
        for layer in self.conv_layer_list:
            seq_embed = layer(seq_embed)

        return self.aggregate_layer(seq_embed)

    def compute_output_shape(self, input_shape):
        batchsize= input_shape[0]
        embed_dim = self.num_filters_list[-1]

        return tf.TensorShape([batchsize, embed_dim])


class BiInteraction(Model):
    def __init__(self, hidden_list, dropout, activation = "relu"):
        super(BiInteraction, self).__init__()
        self.num_dense_layers = len(hidden_list)
        self.activation = activation
        self.dropout= dropout
        self.dense_layer_list = []
        for i in range(self.num_dense_layers):
            self.dense_layer_list.append(Dense(hidden_list[i], activation= activation))
        self.out_layer = Dense(1)


    def call(self, graph_embed, protSeq_embed):
        concat_embed = Concatenate(axis= -1)([graph_embed, protSeq_embed])
        for layer in self.dense_layer_list:
            concat_embed = layer(concat_embed)
            concat_embed = Dropout(self.dropout)(concat_embed)
        return self.out_layer(concat_embed)

    def compute_output_shape(self, input_shape):
        (batchsize, graph_dim), (batchsize, prot_dim) = input_shape
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
