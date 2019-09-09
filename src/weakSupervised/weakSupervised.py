import argparse
from src.weakSupervised.data_utils import load_fn
import tensorflow as tf
from src.weakSupervised.drugPropertyModelSubclass import DrugProperty
from src.weakSupervised.train_utils import loop_dataset
import os
from src.utils import make_config_str, PROJPATH
from pprint import pprint
from src.utils import log
from functools import partial
tf.enable_eager_execution()

def pretrain(drugPropertyModel,
             dataset,
             dir_prefix,
             epoches= 10000,
             batchsize = 128,
             lr = 0.001,
             patience= 8,
             print_every= 32,
             ):
    train, val, test = load_fn(dataset= dataset, batchsize= batchsize)

    chkpt_dir = os.path.join(PROJPATH, "checkpoint/", "weakSuperviesd_%s" % (dir_prefix,))
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)

    log_dir = os.path.join(PROJPATH, "log/", 'weakSupervised_%s' %(dir_prefix, ))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_f = open(os.path.join(log_dir, 'log'), 'w')
    printf = partial(log, f=log_f)
    optimizer = tf.train.AdamOptimizer(learning_rate= lr)

    min_loss = float('inf')
    wait = 0
    for epoch in range(epoches):
        printf("Epoch %d..." %(epoch, ))
        train_loss = loop_dataset(drugPropertyModel, train, optimizer, print_every= print_every)
        printf("Train epoch %d: loss %.4f." %(epoch, train_loss))

        val_loss = loop_dataset(drugPropertyModel, val, print_every= print_every)
        printf("Val epoch %d: loss %.4f." %(epoch, val_loss))

        if val_loss < min_loss:
            min_loss = val_loss
            wait = 0

            drugPropertyModel.save_weights(os.path.join(chkpt_dir, 'drug_property_model'))
        else:
            wait = wait + 1
            if wait == patience:
                printf("early stop at epoch %d" %(epoch, ))
                break

    drugPropertyModel.load_weights(os.path.join(chkpt_dir, 'drug_property_model'))
    test_loss = loop_dataset(drugPropertyModel, test, print_every= print_every)
    printf("Test loss: %.4f" %(test_loss, ))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='kiba_origin', help="dataset to use.")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate in optimizer")
    parser.add_argument("--batchsize", type=int, default=256, help="batchsize during training.")
    parser.add_argument("--atom_hidden", type=int, nargs="+", default=[32, 16],
                        help="atom hidden dimension list in graph embedding model.")
    parser.add_argument("--pair_hidden", type=int, nargs="+", default=[16, 8],
                        help="pair hidden dimension list in graph embedding model.")
    parser.add_argument("--graph_features", type=int, default=32, help="graph features dimension")
    parser.add_argument("--epoches", type=int, default=1, help="epoches during training..")
    parser.add_argument("--patience", type=int, default=1, help="patience epoch to wait during early stopping.")
    parser.add_argument("--print_every", type=int, default=32, help="print intervals during loop dataset.")
    args = parser.parse_args()

    pprint(vars(args))
    configStr = make_config_str(args)
    # configStr = "test"
    atom_dim = 75
    pair_dim = 14
    props_dim = 100

    drugPropertyModel = DrugProperty(atom_features=atom_dim,
                                     pair_features=pair_dim,
                                     atom_hidden_list=args.atom_hidden,
                                     pair_hidden_list=args.pair_hidden,
                                     graph_feat=args.graph_features,
                                     num_mols=args.batchsize,
                                     props_dim=props_dim)
    pretrain(drugPropertyModel,
             args.dataset,
             configStr,
             epoches= args.epoches,
             batchsize= args.batchsize,
             lr= args.lr,
             patience= args.patience,
             print_every= args.print_every,
             )