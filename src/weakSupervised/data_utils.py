import pandas as pd
import os
from src.featurizer import WeaveFeaturizer
from rdkit.Chem import MolFromSmiles
from src.data_utils import gather_mol
import numpy as np
from sklearn.model_selection import train_test_split

def load_fn(dataDir= '../../data/kiba-origin/', batchsize = 32, shuffle = True, seed = 32):
    """

    :param dataDir:
    :param batchsize:
    :param shuffle:
    :param seed:
    :return: iterable train val test dataset
    """
    df = pd.read_csv(os.path.join(dataDir, "kiba_props.csv"))
    prop_array = df[[column for column in df.columns if column != 'smiles']].values
    smiles_list = df['smiles'].tolist()
    train_val_smiles, test_smiles, train_val_props, test_props = train_test_split(smiles_list, prop_array)
    train_smiles, train_props, val_smiles, val_props = train_test_split(train_val_smiles, train_val_props)
    return DataSet(train_smiles, train_props, batchsize, shuffle, seed), \
           DataSet(val_smiles, val_props, batchsize, shuffle, seed), \
           DataSet(test_smiles, test_props, batchsize, shuffle, seed)


class DataSet():
    def __init__(self,
                 smiles_list,
                 prop_array,
                 batchsize,
                 shuffle = True,
                 seed = 32,
                 ):
        """

        :param smiles_list: ar
        :param prop_array:
        :param batchsize:
        :param shuffle:
        :param seed:
        """
        self.featurizer = WeaveFeaturizer()
        self.prop_array = np.array(prop_array)
        self.mol_array = np.array(self.featurizer([MolFromSmiles(smiles) for smiles in smiles_list]))
        self.batchsize= batchsize
        self.seed = seed
        self.shuffle= shuffle


    def __iter__(self,):
        """

        :param batchsize:
        :param inds: a list of inds to use in dataset.
        :param shuffle:
        :param seed:
        :return:
        """
        num_samples = len(self.mol_array)
        sample_inds = np.arange(num_samples)
        num_batches = num_samples // self.batchsize
        rng = np.random.RandomState(self.seed)
        if self.shuffle:
            rng.shuffle(sample_inds)

        for i in range(num_batches):
            batch_inds = sample_inds[ i * self.batchsize : min(num_samples, ( i + 1 ) * self.batchsize)]
            batch_mol = self.mol_array[batch_inds]
            batch_mol_merged = gather_mol(batch_mol)
            batch_props = self.prop_array[batch_inds]
            yield batch_mol_merged, batch_props



if __name__ == '__main__':
    train, val, test = load_fn()
    for i, batch in enumerate(train):
        print(batch)
        if i > 20:
            break


