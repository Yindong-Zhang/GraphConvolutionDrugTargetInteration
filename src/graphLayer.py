import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.python.keras import initializers, activations

class MolecularConvolutionLayer(tf.keras.layers.Layer):
    # TODO:
    def __init__(self,
                 n_atom_input_feat=75,
                 n_pair_input_feat=14,
                 n_atom_output_feat=50,
                 n_pair_output_feat=50,
                 n_atom_agg_feat= 32,
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
        super(MolecularConvolutionLayer, self).__init__(**kwargs)
        self.init = init  # Set weight initialization
        self.activation = activation  # Get activations
        self.update_pair = update_pair  # last weave layer does not need to update
        self.dim_a = n_atom_input_feat
        self.dim_p = n_pair_input_feat
        self.dim_ao = n_atom_output_feat
        self.dim_po = n_pair_output_feat
        self.linear_aa = Dense(n_atom_output_feat, activation=self.activation, use_bias=True, kernel_initializer=self.init)
        self.linear_pa = Dense(n_atom_agg_feat, activation= self.activation, use_bias= True, kernel_initializer= self.init)
        self.linear_ao = Dense(n_atom_output_feat, activation= self.activation, use_bias= True, kernel_initializer= self.init)
        self.linear_ap = Dense(n_pair_output_feat, activation= self.activation, use_bias= True, kernel_initializer= self.init)
        self.linear_pp = Dense(n_pair_output_feat, activation= self.activation, use_bias= True, kernel_initializer= self.init)
        self.bn_pair = BatchNormalization()
        self.bn_atoms = BatchNormalization()

    def call(self, inputs, training= None):
        """Creates weave tensors.

        inputs: atom_features, pair_features, pair_split, atom_to_pair
        """
        atom_features, pair_features, pair_split, atom_to_pair = inputs

        pair_i = atom_to_pair[:, 0]
        pair_j = atom_to_pair[:, 1]

        atom_j = tf.gather(atom_features, pair_j, axis= 0)
        A_paj = self.linear_pa(tf.concat([pair_features, atom_j], axis= -1))
        A_paj_sum = tf.segment_sum(A_paj, pair_i)
        A_pa = self.linear_ao(tf.concat([atom_features, A_paj_sum], axis= -1))
        A_aa = self.linear_aa(atom_features)
        A_s = A_pa + A_aa
        if training:
            A_s = self.bn_atoms(A_s)
        atom_hidden = tf.nn.relu(A_s)

        atom_i = tf.gather(atom_features, pair_i, axis= 0)
        atom_ipj = atom_i + atom_j
        P_apa = self.linear_ap(tf.concat([pair_features, atom_ipj], axis= -1))
        P_pp = self.linear_pp(pair_features)
        P_s = P_apa + P_pp
        if training:
            P_s = self.bn_pair(P_s)
        pair_hidden = tf.nn.relu(P_s)

        return [atom_hidden, pair_hidden]

    def compute_output_shape(self, input_shape):
        atoms_shape, pairs_shape, _, _ = input_shape
        atoms_shape[-1] = self.dim_ao
        pairs_shape[-1] = self.dim_po
        return [tf.TensorShape(atoms_shape), tf.TensorShape(pairs_shape)]


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
        self.dim_aa = n_hidden_AA
        self.dim_pa = n_hidden_PA
        self.dim_ap = n_hidden_AP
        self.dim_pa = n_hidden_PP
        self.n_hidden_A = n_hidden_AA + n_hidden_PA
        self.n_hidden_P = n_hidden_AP + n_hidden_PP

        self.n_atom_input_feat = n_atom_input_feat
        self.n_pair_input_feat = n_pair_input_feat
        self.dim_ao = n_atom_output_feat
        self.dim_po = n_pair_output_feat

    def build(self, input_shape):
        """
            inputs: atom_features, pair_features, pair_split, atom_to_pair
        :param input_shape:
        :return:
        """
        super(WeaveLayer, self).build(input_shape)
        self.linear_aa = Dense(self.n_hidden_A, activation=self.activation, use_bias=True, kernel_initializer=self.init)
        self.linear_pa = Dense(self.dim_pa, activation= self.activation, use_bias= True, kernel_initializer= self.init)
        self.linear_ao = Dense(self.dim_ao, activation= self.activation, use_bias= True, kernel_initializer= self.init)
        self.linear_ap = Dense(self.dim_ap, activation= self.activation, use_bias= True, kernel_initializer= self.init)
        self.linear_pp = Dense(self.dim_pa, activation= self.activation, use_bias= True, kernel_initializer= self.init)
        self.linear_po = Dense(self.dim_po, activation= self.activation, use_bias= True, kernel_initializer= self.init)

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
        atoms_shape[-1] = self.dim_ao
        pairs_shape[-1] = self.dim_po
        return [tf.TensorShape(atoms_shape), tf.TensorShape(pairs_shape)]


class WeaveGather(tf.keras.layers.Layer):

    def __init__(self,
                 num_mols,
                 atom_dim=128,
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
        atom_dim: int, optional
          number of features for each input molecule
        gaussian_expand: boolean. optional
          Whether to expand each dimension of atomic features by gaussian histogram
        init: str, optional
          Weight initialization for filters.
        activation: str, optional
          Activation function applied
        """
        super(WeaveGather, self).__init__(**kwargs)
        self.n_input = atom_dim
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




