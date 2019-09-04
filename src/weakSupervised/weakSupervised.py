from src.model_subclass import GraphEmbedding
import argparse
from src.weakSupervised.data_utils import create_dataset

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

atom_dim = 75
pair_dim = 14
graph_embedding_model = GraphEmbedding(atom_features= atom_dim,
                                       pair_features= pair_dim,
                                       atom_hidden_list= args.atom_hidden,
                                       pair_hidden_list= args.pair_hidden,
                                       graph_feat= args.atom_hidden[-1],
                                       num_mols= args.batchsize)


dataset = create_dataset(args.batchsize)

for epoch in range(args.epoches):
    for batch in dataset:
        graph_embedding = graph_embedding_model()