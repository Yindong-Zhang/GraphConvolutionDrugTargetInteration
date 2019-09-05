from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense
from src.model_subclass import GraphEmbedding
import tensorflow as tf
class DrugProperty(Model):
    def __init__(self,
                 atom_features,
                 pair_features,
                 atom_hidden_list,
                 pair_hidden_list,
                 graph_feat,
                 num_mols,
                 props_dim,
                 ):
        super(DrugProperty, self).__init__(self, )
        self.graph_embedding_model = GraphEmbedding(atom_features= atom_features,
                                               pair_features= pair_features,
                                               atom_hidden_list=atom_hidden_list,
                                               pair_hidden_list= pair_hidden_list,
                                               graph_feat= graph_feat,
                                               num_mols= num_mols)
        self.props_dim = props_dim
        self.dense_layer = Dense(props_dim)

    def call(self, mols):
        graph_embedding = self.graph_embedding_model(mols)
        props_pred = self.dense_layer(graph_embedding)
        return props_pred

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([self.num_mols, self.props_dim])