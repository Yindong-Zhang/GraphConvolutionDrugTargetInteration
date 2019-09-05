from src.model_subclass import GraphEmbedding
import argparse
from src.weakSupervised.data_utils import load_fn
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.losses import mean_squared_error
import os
from src.utils import make_config_str
from pprint import pprint
tf.enable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type= float, default= 0.001, help= "learning rate in optimizer")
parser.add_argument("--batchsize", type= int, default= 32, help = "batchsize during training.")
parser.add_argument("--atom_hidden", type= int, nargs= "+", default= [32, 16], help= "atom hidden dimension list in graph embedding model.")
parser.add_argument("--pair_hidden", type= int, nargs= "+", default= [16, 8], help = "pair hidden dimension list in graph embedding model.")
parser.add_argument("--graph_features", type= int, default= 32, help= "graph features dimension")
parser.add_argument("--epoches", type= int, default= 2, help= "epoches during training..")
parser.add_argument("--patience", type= int, default= 1, help= "patience epoch to wait during early stopping.")
parser.add_argument("--print_every", type= int, default= 32, help= "print intervals during loop dataset.")
args = parser.parse_args()

pprint(vars(args))
configStr = make_config_str(args)
# configStr = "test"
chkpt_dir = os.path.join("../../checkpoint", configStr)
atom_dim = 75
pair_dim = 14
props_dim = 100

graph_embedding_model = GraphEmbedding(atom_features= atom_dim,
                                       pair_features= pair_dim,
                                       atom_hidden_list= args.atom_hidden,
                                       pair_hidden_list= args.pair_hidden,
                                       graph_feat= args.atom_hidden[-1],
                                       num_mols= args.batchsize)
dense_layer = Dense(props_dim)

train, val, test = load_fn(batchsize= args.batchsize)

def loop_dataset(dataset, optimizer= None):
    mean_loss = 0
    for it, (mols, props) in enumerate(dataset):
        with tf.GradientTape() as tape:
            graph_embedding = graph_embedding_model(mols)
            props_pred = (graph_embedding)
            loss = mean_squared_error(props_pred, props)
        mean_loss = (it * mean_loss + loss) / (it + 1)

        if optimizer:
            variables = graph_embedding_model.variables + Dense.variables
            grads = tape.gradient(loss, variables)
            optimizer.apply_gradient(zip(grads, variables))

        if it % args.print_every == 0:
            print("%d: loss %.4f." %(it, mean_loss))
    return mean_loss

optimizer = tf.train.AdamOptimizer(learning_rate= args.lr)

min_loss = float('inf')
wait = 0
for epoch in range(args.epoches):
    print("Epoch %d..." %(epoch, ))
    train_loss = loop_dataset(train, optimizer)
    print("Train epoch %d: loss %.4f." %(epoch, train_loss))

    val_loss = loop_dataset(val)
    print("Val epoch %d: loss %.4f." %(epoch, val_loss))

    if val_loss < min_loss:
        min_loss = val_loss
        wait = 0

        graph_embedding_model.save_weights(os.path.join(chkpt_dir, 'graph_embedding_model'))
        dense_layer.save_weights(os.path.join(chkpt_dir, 'dense_layer'))
    else:
        wait = wait + 1;
        if wait == args.patience:
            print("early stop at epoch %d" %(epoch, ))
            break

graph_embedding_model.load_weights(os.path.join(chkpt_dir, 'graph_embedding_model'))
dense_layer.load_weights(os.path.join(chkpt_dir, 'dense_layer'))
test_loss = loop_dataset(test)
print("Test loss: %.4f" %(test_loss, ))