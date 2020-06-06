import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, BatchNormalization, Embedding, Conv1D, MaxPooling1D, Concatenate, Dropout, Layer, LSTM, Bidirectional, Masking
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras import initializers
from src.graphLayer import WeaveLayer, WeaveGather, MolecularConvolutionLayer

""" model subclass is more suitable for tensorflow 2.0
"""
class GraphEmbedding(Model):
    def __init__(self, atom_features, pair_features, atom_hidden_list, pair_hidden_list, graph_feat, num_mols,
                 dropout = 0.1, weight_decay = 1E05, residual_connection= True, leaky_slope = 0.1, *args, **kwargs):
        super(GraphEmbedding, self).__init__(*args, **kwargs)
        self.atom_hidden_list = atom_hidden_list
        self.pair_hidden_list = pair_hidden_list
        self.residual_connection = residual_connection
        self.activation = LeakyReLU(leaky_slope)
        self.graph_features = graph_feat
        self.num_mols = num_mols
        self.weight_decay = weight_decay
        assert len(atom_hidden_list) == len(pair_hidden_list), "length of atom hidden list should equal to length of pair hidden list."
        self.num_GCNLayers = len(atom_hidden_list)

        self.atom_features = atom_features
        self.pair_features = pair_features

        self.GCLayer_list = []

        atom_dim_sum = atom_features
        pair_dim_sum = pair_features
        self.atom_hidden_input_list = [atom_dim_sum, ]
        self.pair_hidden_input_list = [pair_dim_sum, ]
        if self.residual_connection:
            for i in range(self.num_GCNLayers):
                atom_dim_sum = atom_dim_sum + self.atom_hidden_list[i]
                pair_dim_sum = pair_dim_sum + self.pair_hidden_list[i]
                self.atom_hidden_input_list.append(atom_dim_sum)
                self.pair_hidden_input_list.append(pair_dim_sum)
        else:
            for i in range(self.num_GCNLayers - 1):
                self.atom_hidden_input_list.append(self.atom_hidden_list[i])
                self.pair_hidden_input_list.append(self.pair_hidden_list[i])
        for i in range(self.num_GCNLayers):
            # self.GCLayer_list.append(
            #     MolecularConvolutionLayer(self.atom_hidden_input_list[i], self.pair_hidden_input_list[i], self.atom_hidden_list[i], self.pair_hidden_list[i], self.atom_hidden_list[i]))
            self.GCLayer_list.append(
                WeaveLayer(self.atom_hidden_input_list[i], self.pair_hidden_input_list[i], self.atom_hidden_list[i], self.pair_hidden_list[i], weight_decay = self.weight_decay, activation= self.activation))

        self.dropout_layer = Dropout(dropout)
        self.dense = Dense(self.graph_features, activation= self.activation, kernel_regularizer= l2(self.weight_decay))

    def call(self, inputs, training= None):
        atom_features, pair_features, pair_split, atom_split, atom_to_pair, num_atoms = inputs
        atom_hidden_list = []
        pair_hidden_list = []
        atom_hidden, pair_hidden = atom_features, pair_features

        atom_hidden_list.append(atom_hidden)
        pair_hidden_list.append(pair_hidden)
        for i in range(self.num_GCNLayers):
            if self.residual_connection:
                atom_hidden = tf.concat(atom_hidden_list, axis= -1)
                pair_hidden = tf.concat(pair_hidden_list, axis= -1)
            atom_hidden = self.dropout_layer(atom_hidden, training = training)
            pair_hidden = self.dropout_layer(pair_hidden, training= training)
            atom_hidden, pair_hidden = self.GCLayer_list[i]([atom_hidden, pair_hidden, pair_split, atom_to_pair, num_atoms])
            atom_hidden_list.append(atom_hidden)
            pair_hidden_list.append(pair_hidden)

        if self.residual_connection:
            atom_hidden_out = tf.concat(atom_hidden_list, axis= -1) # use dense connection at the last dense layer
            # print(" training: %s" %(training, ))
            atom_hidden = self.dense(atom_hidden_out)
        else:
            atom_hidden = self.dense(atom_hidden)

        # print(atom_hidden[:5])
        return atom_hidden

    def compute_output_shape(self, input_shape):
        atom_shape, _, _, _, _ = input_shape
        num_atoms, atom_dim = atom_shape
        return tf.TensorShape([num_atoms, self.graph_features])

class ProtSeqEmbedding(Model):
    def __init__(self, num_filters_list, filter_length_list, prot_char_size, max_seq_length, leaky_slope= 0.1, **kwargs):
        super(ProtSeqEmbedding, self).__init__(**kwargs)
        self.prot_seq_length = prot_char_size
        self.max_seq_length = max_seq_length
        self.activation = LeakyReLU(leaky_slope)
        assert len(num_filters_list) == len(filter_length_list), "Incompatibe hyperparameters."
        self.num_conv_layers = len(num_filters_list)
        self.num_filters_list = num_filters_list
        self.filter_length_list = filter_length_list
        self.embed =  Embedding(input_dim=prot_char_size + 1, output_dim=128, input_length= max_seq_length, mask_zero= True)
        self.rnn_layer_0 = Bidirectional(LSTM(128, return_sequences= True))
        self.rnn_layer_1 = Bidirectional(LSTM(128, return_sequences= True))
        self.conv_layer_list = []
        for i in range(self.num_conv_layers):
            self.conv_layer_list.append(Conv1D(filters= num_filters_list[i],
                                                     kernel_size= filter_length_list[i],
                                                     activation= self.activation,
                                                     padding='same',
                                                     strides= 1))

        self.maxpool1d = MaxPooling1D(2, strides= 2)

    def call(self, inputs):
        seq_embed= self.embed(inputs)
        seq_embed = self.rnn_layer_0(seq_embed)
        seq_embed = self.rnn_layer_1(seq_embed)
        for layer in self.conv_layer_list:
            seq_embed = layer(seq_embed)
            # seq_embed = self.maxpool1d(seq_embed)
        # print(seq_embed[:5])
        return seq_embed

    def compute_output_shape(self, input_shape):
        batchsize, protSeqLen = input_shape
        embed_dim = self.num_filters_list[-1]

        return tf.TensorShape([batchsize, protSeqLen, embed_dim])

class EmbeddingLayer(Layer):
    def __init__(self, size_ls, embed_size_ls, **kwargs): # kwargs, 申明可变参数
        """

        :param size_dict: a ls indicate [feat_dim, ]
        """
        super(EmbeddingLayer, self).__init__(**kwargs)
        assert len(size_ls) == len(embed_size_ls), "Inconsistent Arguments"
        self.dim_ls = size_ls
        self.embed_size_ls= embed_size_ls
        self.embed_layers = []
        self.num_feat = len(size_ls)
        for i in range(self.num_feat):
            self.embed_layers.append(Embedding(size_ls[i], embed_size_ls[i]))

    def call(self, inputs, **kwargs):
        """

        :param inputs: should be a list of feat array in format {feat_name: data array}
        :param kwargs:
        :return:
        """
        embed_ls = []
        for i, data in enumerate(inputs):
            embed_ls.append(self.embed_layers[i](data))

        ret = tf.concat(embed_ls, axis = -1)
        return ret

    def compute_output_shape(self, input_shape):
        (b, ) = input_shape[0]
        out_dim = sum(self.embed_size_ls)
        return tf.TensorShape([b, out_dim])



class BiInteraction(Layer):
    def __init__(self, hidden_list, dropout, activation = "relu", initializer = 'he_uniform', weight_decay= 1E-6, protSeq_max_length = 1000, **kwargs):
        super(BiInteraction, self).__init__(**kwargs)
        self.num_dense_layers = len(hidden_list)
        self.activation = activation
        self.initializer = initializer
        self.weight_decay = weight_decay
        self.PROTSEQ_MAX_LEN= protSeq_max_length
        self.dropout= dropout
        self.dense_layer_list = []
        for i in range(self.num_dense_layers):
            self.dense_layer_list.append(Dense(hidden_list[i], activation= activation, kernel_initializer= self.initializer, kernel_regularizer= l2(self.weight_decay)))
        self.out_layer = Dense(1)
        self.dropout_layer = Dropout(self.dropout)

    def build(self, input_shape):
        atom_hidden_shape, prot_hidden_shape, atom_splits_shape, prot_len_shape = input_shape
        atom_dim = atom_hidden_shape[-1]
        prot_dim = prot_hidden_shape[-1]
        self.W= self.add_weight('attention_weight', shape= (atom_dim, prot_dim), initializer= initializers.get(self.initializer),
                                regularizer= l2(self.weight_decay))


    def call(self, inputs, training= None):
        atom_embed, protSeq_embed, atom_splits, protSeq_len = inputs
        # print(atom_embed[:5])
        # print(protSeq_embed[0, :, :2])
        protSeq_embed_T = tf.transpose(protSeq_embed, (0, 2, 1))
        protSeq_embed_gather = tf.gather(protSeq_embed_T, atom_splits, axis= 0)
        protSeq_mask = tf.sequence_mask(protSeq_len, self.PROTSEQ_MAX_LEN)
        protSeq_mask_gather = tf.gather(protSeq_mask, atom_splits, axis = 0) # shape (num_atom, len_protSeq)
        W = tf.einsum('ij, jk, ikl->il', atom_embed, self.W, protSeq_embed_gather)
        # print(W[0])
        W = tf.tanh(W)
        E = -9E15 * tf.ones_like(W)
        W = tf.where(protSeq_mask_gather, W, E)
        # print(W[0])
        Wc = tf.exp(tf.reduce_max(W, axis= -1 ,keepdims= True))
        Sc = tf.gather(tf.segment_sum(Wc, atom_splits), atom_splits, axis= 0)
        aa = Wc / Sc
        # print(aa[:100])
        atom_embed = tf.segment_sum(tf.multiply(aa, atom_embed), atom_splits)

        Wp = tf.segment_max(W, atom_splits)
        ap = tf.nn.softmax(Wp, axis= -1)

        # print(ap[0])
        prot_embed = tf.einsum('ij, ijk->ik', ap, protSeq_embed)

        concat_embed = tf.concat([atom_embed, prot_embed], axis = -1)
        for layer in self.dense_layer_list:
            concat_embed = layer(concat_embed)
            # concat_embed = BatchNormalization(axis = -1)(concat_embed, training= training)
            concat_embed = self.dropout_layer(concat_embed, training= training) # reuse is a problem ? ok for dropout layer but not for batch normalization layer?
        return self.out_layer(concat_embed)

    def compute_output_shape(self, input_shape):
        (num_atoms, atom_dim), (batchsize, prot_dim), (num_atoms, ), (batchsize, ) = input_shape
        return tf.TensorShape((batchsize, 1))

class ConcatBiInteraction(Layer):
    def __init__(self, hidden_list, dropout, activation = "relu", initializer = 'he_uniform', weight_decay = 1E-6, **kwargs):
        super(ConcatBiInteraction, self).__init__(**kwargs)
        self.num_dense_layers = len(hidden_list)
        self.activation = activation
        self.weight_decay = weight_decay
        self.initializer = initializer
        self.dropout= dropout
        self.att_layer_1 = Dense(128, use_bias= True, activation = 'tanh')
        self.att_layer_2 = Dense(1, activation = 'tanh') # activation = tanh ?
        self.dense_layer_list = []
        for i in range(self.num_dense_layers):
            self.dense_layer_list.append(Dense(hidden_list[i], activation= activation, kernel_initializer= self.initializer,
                                               kernel_regularizer= l2(self.weight_decay)))

        self.out_layer = Dense(1)
        self.dropout_layer = Dropout(self.dropout)

    def call(self, inputs, training= None):
        atom_embed, protSeq_embed, atom_splits = inputs

        protSeq_len = protSeq_embed.shape[1] # protSeq inshape (batchsize, seqlen, embed_dim)
        # print(protSeq_embed[0, :, :2])
        # print(atom_embed[:10])
        protSeq_embed_gather = tf.gather(protSeq_embed, atom_splits, axis= 0)
        atom_embed_expand = tf.tile(tf.expand_dims(atom_embed, 1), [1, protSeq_len, 1]) # atom_embed (n_atom, d_atom) to ( (n_atom, protSeq_len, d_atom)
        concat_embed = tf.concat([protSeq_embed_gather, atom_embed_expand], axis = -1)
        concat_hidden = self.att_layer_1(concat_embed)
        W = self.att_layer_2(concat_hidden)
        print(W[:2])
        W = tf.squeeze(W, axis= -1) # to reduce the last 1 dimension

        W = 5 * W
        Wc = tf.exp(tf.reduce_max(W, axis= -1 ,keepdims= True))
        Sc = tf.gather(tf.segment_sum(Wc, atom_splits), atom_splits, axis= 0)
        aa = Wc / Sc
        # print(aa[:200])
        atom_embed = tf.segment_sum(aa * atom_embed, atom_splits)

        # print(W[0])
        Wp = tf.segment_max(W, atom_splits)
        # print(Wp[0])
        ap = tf.nn.softmax(Wp, axis= -1)
        # print(ap.shape, protSeq_embed.shape)
        prot_embed = tf.einsum('ij, ijk->ik', ap, protSeq_embed)

        concat_embed = tf.concat([atom_embed, prot_embed], axis = -1)
        for layer in self.dense_layer_list:
            concat_embed = layer(concat_embed)
            # concat_embed = BatchNormalization(axis = -1)(concat_embed, training= training)
            concat_embed = self.dropout_layer(concat_embed, training= training)

        return self.out_layer(concat_embed)

    def compute_output_shape(self, input_shape):
        (num_atoms, atom_dim), (batchsize, prot_dim), (num_atoms, ) = input_shape
        return tf.TensorShape((batchsize, 1))

class ConcatMlp(Layer):
    def __init__(self, hidden_list= [512, 1024], dropout= 0.1, weight_decay= 1E-5, activation ='relu', initializer ='he_uniform',  **kwargs):
        super(ConcatMlp, self).__init__(**kwargs)

        self.hidden_size= hidden_list
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.activation = activation
        self.initializer = initializer

    def build(self, input_shape):
        self.hidden_layers =[]
        for hidden in self.hidden_size:
            self.hidden_layers.append(Dense(hidden, activation= self.activation, kernel_initializer= self.initializer,
                                            kernel_regularizer= l2(self.weight_decay)))
        self.output_layer = Dense(1)
        self.dropout_layer = Dropout(self.dropout)

    def call(self, inputs, training= None):
        """

        :param inputs: [atom_embed, protSeq_embed, atom_split] in shape ( n_atoms, atom_hidden), (batchsize, seqlen, hidden), (n_atoms, )
        :return:
        """
        atom_embed, protSeq_embed, atom_split = inputs
        mol_embed = tf.segment_max(atom_embed, atom_split)
        prot_embed = tf.reduce_max(protSeq_embed, axis = 1)
        hidden = tf.concat([mol_embed, prot_embed], axis = -1)
        for layer in self.hidden_layers:
            hidden = layer(hidden)
            # hidden = BatchNormalization(axis= -1)(hidden, training = training)
            hidden = self.dropout_layer(hidden, training= training)

        output = self.output_layer(hidden)
        return output

if __name__ == "__main__":
    tf.enable_eager_execution()

    # graph_embed_model = GraphEmbedding([8, 8], [4, 4], 16, 8)
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
    # graph_embed_model.summary()
    # concatMlp = ConcatMlp()
    # concatMlp.build([tf.TensorShape((100, 12)),
    #                  tf.TensorShape((32, 100, 12)),
    #                  tf.TensorShape((100, ))])
    # concatMlp.compile(optimizer = 'adam', loss = 'mse')
    # concatMlp.summary()
    # atom_embed = tf.random((100, ))
