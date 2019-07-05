import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, BatchNormalization
from src.graphLayer import WeaveLayer, WeaveGather

class GraphEmbedding(Model):
    def __init__(self, atom_feat, pair_feat, atom_hidden_list, pair_hidden_list, graph_feat, num_mols):
        self.atom_features = atom_feat
        self.pair_features = pair_feat
        self.atom_hidden_list = atom_hidden_list
        self.pair_hidden_list = pair_hidden_list
        self.graph_features = graph_feat
        self.num_mols = num_mols
        assert len(atom_hidden_list) == len(pair_hidden_list), "length of atom hidden list should equal to length of pair hidden list."
        self.num_weaveLayers = len(atom_hidden_list)
        self.weaveLayer_list = []

        self.weaveLayer_list.append(WeaveLayer(atom_feat, pair_feat, atom_hidden_list[0], pair_hidden_list[0]))
        for i in range(1, self.num_weaveLayers):
            self.weaveLayer_list.append(WeaveLayer(atom_hidden_list[i - 1], pair_hidden_list[i - 1], atom_hidden_list[i], pair_hidden_list[i]))

        self.dense = Dense(graph_feat, activation= 'tanh')
        self.batchnorm = BatchNormalization()
        self.weavegather = WeaveGather(num_mols, self.graph_features, guassian_expand = True)

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

class BiInteraction(Model):
    def __init__(self, num_filters, filter_length1, filter_length2, max_seq_length):
        super(BiInteraction, self).__init__()
        self.num_filters = num_filters
        self.filter_length1 = filter_length1
        self.filter_length2 = filter_length2
        self.max_seq_length = max_seq_length

        self.graph_embedding = GraphEmbedding()

    def call(self, input):
        # TODO: