import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import initializers, activations

class WeaveLayer(tf.keras.layers.Layer):

    def __init__(self,
                 n_atom_input_feat=75,
                 n_pair_input_feat=14,
                 n_atom_output_feat=50,
                 n_pair_output_feat=50,
                 n_hidden_AA=50,
                 n_hidden_PA=50,
                 n_hidden_AP=50,
                 n_hidden_PP=50,
                 update_pair=True,
                 init='glorot_uniform',
                 activation='relu',
                 **kwargs):
        """
        Parameters
        ----------
        n_atom_input_feat: int, optional
          Number of features for each atom in input.
        n_pair_input_feat: int, optional
          Number of features for each pair of atoms in input.
        n_atom_output_feat: int, optional
          Number of features for each atom in output.
        n_pair_output_feat: int, optional
          Number of features for each pair of atoms in output.
        n_hidden_XX: int, optional
          Number of units(convolution depths) in corresponding hidden layer
        update_pair: bool, optional
          Whether to calculate for pair features,
          could be turned off for last layer
        init: str, optional
          Weight initialization for filters.
        activation: str, optional
          Activation function applied
        """
        super(WeaveLayer, self).__init__(**kwargs)
        self.init = init  # Set weight initialization
        self.activation = activation  # Get activations
        self.update_pair = update_pair  # last weave layer does not need to update
        self.n_hidden_AA = n_hidden_AA
        self.n_hidden_PA = n_hidden_PA
        self.n_hidden_AP = n_hidden_AP
        self.n_hidden_PP = n_hidden_PP
        self.n_hidden_A = n_hidden_AA + n_hidden_PA
        self.n_hidden_P = n_hidden_AP + n_hidden_PP

        self.n_atom_input_feat = n_atom_input_feat
        self.n_pair_input_feat = n_pair_input_feat
        self.n_atom_output_feat = n_atom_output_feat
        self.n_pair_output_feat = n_pair_output_feat
        # self.W_AP, self.b_AP, self.W_PP, self.b_PP, self.W_P, self.b_P = None, None, None, None, None, None

    def build(self, input_shape):
        """
            inputs: atom_features, pair_features, pair_split, atom_to_pair
        :param input_shape:
        :return:
        """
        self.linear_aa = Dense(self.n_hidden_A, activation=self.activation, use_bias=True, kernel_initializer=self.init)
        # self.W_AA = self.add_weight("W_AA", shape= [self.n_atom_input_feat, self.n_hidden_AA], initializer= self.init)
        # self.W_AA = init([self.n_atom_input_feat, self.n_hidden_AA])
        # self.b_AA = model_ops.zeros(shape=[
        #     self.n_hidden_AA,
        # ])
        self.linear_pa = Dense(self.n_hidden_PA, activation= self.activation, use_bias= True, kernel_initializer= self.init)
        # self.W_PA = init([self.n_pair_input_feat, self.n_hidden_PA])
        # self.b_PA = model_ops.zeros(shape=[
        #   self.n_hidden_PA,
        # ])

        self.linear_ao = Dense(self.n_atom_output_feat, activation= self.activation, use_bias= True, kernel_initializer= self.init)
        # self.W_A = init([self.n_hidden_A, self.n_atom_output_feat])
        # self.b_A = model_ops.zeros(shape=[
        #   self.n_atom_output_feat,
        # ])

        self.linear_ap = Dense(self.n_hidden_AP, activation= self.activation, use_bias= True, kernel_initializer= self.init)
        # self.W_AP = init([self.n_atom_input_feat * 2, self.n_hidden_AP])
        # self.b_AP = model_ops.zeros(shape=[
        #   self.n_hidden_AP,
        # ])

        self.linear_pp = Dense(self.n_hidden_PP, activation= self.activation, use_bias= True, kernel_initializer= self.init)
        # self.W_PP = init([self.n_pair_input_feat, self.n_hidden_PP])
        # self.b_PP = model_ops.zeros(shape=[
        #   self.n_hidden_PP,
        # ])

        self.linear_po = Dense(self.n_pair_output_feat, activation= self.activation, use_bias= True, kernel_initializer= self.init)

        # self.W_P = init([self.n_hidden_P, self.n_pair_output_feat])
        # self.b_P = model_ops.zeros(shape=[
        #   self.n_pair_output_feat,
        # ])
        super(WeaveLayer, self).build(input_shape)

    def call(self, inputs):
        """Creates weave tensors.

        inputs: atom_features, pair_features, pair_split, atom_to_pair
        """
        atom_features, pair_features, pair_split, atom_to_pair = inputs

        AA = self.linear_aa(atom_features)
        PA = self.linear_pa(pair_features)
        PA = tf.segment_sum(PA, pair_split)

        atom_hidden = tf.concat([AA, PA], -1)
        A = self.linear_ao(atom_hidden)
        # A = tf.matmul(tf.concat([AA, PA], 1), self.W_A) + self.b_A
        # A = activation(A)

        if self.update_pair:
            # AP_ij = tf.matmul(
            #   tf.reshape(
            #     tf.gather(atom_features, atom_to_pair),
            #     [-1, 2 * self.n_atom_input_feat]), self.W_AP) + self.b_AP
            # AP_ij = activation(AP_ij)
            AP_ij = self.linear_ap(tf.reshape(
                tf.gather(atom_features, atom_to_pair),
                [-1, 2 * self.n_atom_input_feat]))

            AP_ji = self.linear_ap(tf.reshape(tf.gather(atom_features, tf.reverse(atom_to_pair, [1])),
                                              [-1, 2 * self.n_atom_input_feat]))
            # AP_ji = tf.matmul(
            #   tf.reshape(
            #     tf.gather(atom_features, tf.reverse(atom_to_pair, [1])),
            #     [-1, 2 * self.n_atom_input_feat]), self.W_AP) + self.b_AP
            # AP_ji = activation(AP_ji)

            PP = self.linear_pp(pair_features)
            # PP = tf.matmul(pair_features, self.W_PP) + self.b_PP
            # PP = activation(PP)

            P = self.linear_po(tf.concat([AP_ij + AP_ji, PP], -1))
            # P = tf.matmul(tf.concat([AP_ij + AP_ji, PP], 1), self.W_P) + self.b_P
            # P = activation(P)
        else:
            P = pair_features

        return [A, P]

    def compute_output_shape(self, input_shape):
        atoms_shape, pairs_shape, _, _ = input_shape
        atoms_shape[-1] = self.n_atom_output_feat
        pairs_shape[-1] = self.n_pair_output_feat
        return [tf.TensorShape(atoms_shape), tf.TensorShape(pairs_shape)]


class WeaveGather(tf.keras.layers.Layer):

    def __init__(self,
                 num_mols,
                 n_input=128,
                 gaussian_expand= True,
                 init='glorot_uniform',
                 activation='tanh',
                 epsilon=1e-3,
                 momentum=0.99,
                 **kwargs):
        """
        Parameters
        ----------
        num_mols: int
          number of molecules in a batch
        n_input: int, optional
          number of features for each input molecule
        gaussian_expand: boolean. optional
          Whether to expand each dimension of atomic features by gaussian histogram
        init: str, optional
          Weight initialization for filters.
        activation: str, optional
          Activation function applied
        """
        super(WeaveGather, self).__init__(**kwargs)
        self.n_input = n_input
        self.batch_size = num_mols
        self.gaussian_expand = gaussian_expand
        self.init = initializers.get(init)  # Set weight initialization
        self.activation = activations.get(activation)  # Get activations
        self.epsilon = epsilon
        self.momentum = momentum

    def build(self, input_shape):
        if self.gaussian_expand:
            self.W = self.add_weight("weight", [self.n_input * 11, self.n_input], initializer = self.init)
            self.b = self.add_weight("bias", shape=[self.n_input, ], initializer = initializers.get("zeros"))
        self.built = True

        super(WeaveGather, self).build(input_shape)

    def call(self, inputs):
        outputs, atom_split = inputs

        if self.gaussian_expand:
            outputs = self.gaussian_histogram(outputs)

        output_molecules = tf.segment_sum(outputs, atom_split)

        if self.gaussian_expand:
            output_molecules = tf.matmul(output_molecules, self.W) + self.b
            output_molecules = self.activation(output_molecules)

        return output_molecules

    def gaussian_histogram(self, x):
        gaussian_memberships = [(-1.645, 0.283), (-1.080, 0.170), (-0.739, 0.134),
                                (-0.468, 0.118), (-0.228, 0.114), (0., 0.114),
                                (0.228, 0.114), (0.468, 0.118), (0.739, 0.134),
                                (1.080, 0.170), (1.645, 0.283)]
        dist = [
            tf.distributions.Normal(p[0], p[1])
            for p in gaussian_memberships
        ]
        dist_max = [dist[i].prob(gaussian_memberships[i][0]) for i in range(11)]
        outputs = [dist[i].prob(x) / dist_max[i] for i in range(11)]
        outputs = tf.stack(outputs, axis=2)
        outputs = outputs / tf.reduce_sum(outputs, axis=2, keepdims=True)
        outputs = tf.reshape(outputs, [-1, self.n_input * 11])
        return outputs

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape([self.batch_size, self.n_input])

class WeightedGather(tf.keras.layers.Layer):
    def __init__(self, ):
        pass


    def call(self, inputs):

        atom_features, atom_split, protSeq_features = inputs


