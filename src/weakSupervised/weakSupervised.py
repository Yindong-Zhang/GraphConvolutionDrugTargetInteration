
import argparse
from src.weakSupervised.data_utils import load_fn
import tensorflow as tf
from src.weakSupervised.drugPropertyModelSubclass import DrugProperty
from src.weakSupervised.train_utils import loop_dataset
import os
from src.utils import make_config_str, PROJPATH
from pprint import pprint
tf.enable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type= str, default= 'kiba', help= "dataset to use.")
parser.add_argument("--lr", type= float, default= 0.001, help= "learning rate in optimizer")
parser.add_argument("--batchsize", type= int, default= 256, help = "batchsize during training.")
parser.add_argument("--atom_hidden", type= int, nargs= "+", default= [32, 16], help= "atom hidden dimension list in graph embedding model.")
parser.add_argument("--pair_hidden", type= int, nargs= "+", default= [16, 8], help = "pair hidden dimension list in graph embedding model.")
parser.add_argument("--graph_features", type= int, default= 32, help= "graph features dimension")
parser.add_argument("--epoches", type= int, default= 1, help= "epoches during training..")
parser.add_argument("--patience", type= int, default= 1, help= "patience epoch to wait during early stopping.")
parser.add_argument("--print_every", type= int, default= 32, help= "print intervals during loop dataset.")
args = parser.parse_args()

pprint(vars(args))
configStr = make_config_str(args)
# configStr = "test"
chkpt_dir = os.path.join(PROJPATH, "checkpoint/", configStr)
atom_dim = 75
pair_dim = 14
props_dim = 100

model = DrugProperty(atom_features= atom_dim,
                     pair_features= pair_dim,
                     atom_hidden_list= args.atom_hidden,
                     pair_hidden_list= args.pair_hidden,
                     graph_feat= args.graph_features,
                     num_mols= args.batchsize,
                     props_dim= props_dim)


train, val, test = load_fn(dataset= args.dataset, batchsize= args.batchsize)

optimizer = tf.train.AdamOptimizer(learning_rate= args.lr)

min_loss = float('inf')
wait = 0
for epoch in range(args.epoches):
    print("Epoch %d..." %(epoch, ))
    train_loss = loop_dataset(train, optimizer, print_every= args.print_every)
    print("Train epoch %d: loss %.4f." %(epoch, train_loss))

    val_loss = loop_dataset(val, print_every= args.print_every)
    print("Val epoch %d: loss %.4f." %(epoch, val_loss))

    if val_loss < min_loss:
        min_loss = val_loss
        wait = 0

        model.save_weights(os.path.join(chkpt_dir, 'drug_property_model'))
    else:
        wait = wait + 1
        if wait == args.patience:
            print("early stop at epoch %d" %(epoch, ))
            break

model.load_weights(os.path.join(chkpt_dir, 'drug_property_model'))
test_loss = loop_dataset(test, print_every= args.print_every)
print("Test loss: %.4f" %(test_loss, ))