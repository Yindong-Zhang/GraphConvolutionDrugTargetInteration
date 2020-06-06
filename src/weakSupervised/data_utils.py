import pandas as pd
import os
from src.featurizer import WeaveFeaturizer
from rdkit.Chem import MolFromSmiles
from src.data_utils import gather_mol
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
from src.utils import PROJPATH
from time import  time
NUMPROCESS = 32

def load_fn(dataset= 'kiba_origin', batchsize = 32, shuffle = True, seed = 32):
    """

    :param dataDir:
    :param batchsize:
    :param shuffle:
    :param seed:
    :return: iterable train val test dataset
    """
    dataDir = os.path.join(PROJPATH, 'data/kiba-origin/')
    df = pd.read_csv(os.path.join(dataDir, "%s_props.csv" %(dataset, ))) # 52077 moleculars, 100 dim property
    prop_array = df[[column for column in df.columns if column != 'smiles']].values

    standardizer = StandardScaler()
    prop_array = standardizer.fit_transform(prop_array)
    smiles_list = df['smiles'].tolist()
    train_val_smiles, test_smiles, train_val_props, test_props = train_test_split(smiles_list, prop_array, test_size= 0.2)
    train_smiles, val_smiles, train_props, val_props = train_test_split(train_val_smiles, train_val_props, test_size= 0.1)
    return DataSet(train_smiles, train_props, batchsize, shuffle, seed), \
           DataSet(val_smiles, val_props, batchsize, shuffle, seed), \
           DataSet(test_smiles, test_props, batchsize, shuffle, seed)

def featurizer(smiles):
    featurizer = WeaveFeaturizer()
    return featurizer([MolFromSmiles(smiles), ])[0]

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
        self.prop_array = np.array(prop_array)
        self.smiles_list = smiles_list
        self.num_samples = len(smiles_list)
        pool = Pool(NUMPROCESS)
        t0 = time()
        apply_res = [pool.apply_async(featurizer, (smiles, )) for smiles in smiles_list]
        self.mol_list = [res.get() for res in apply_res]
        t1 = time()
        print('molecules featurized in %.2f seconds' %(t1 - t0, ))
        pool.close()

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
        sample_inds = np.arange(self.num_samples)
        num_batches = self.num_samples // self.batchsize
        rng = np.random.RandomState(self.seed)
        if self.shuffle:
            rng.shuffle(sample_inds)

        for i in range(num_batches):
            batch_inds = sample_inds[ i * self.batchsize : min(self.num_samples, ( i + 1 ) * self.batchsize)]
            batch_mol = [self.mol_list[ind] for ind in batch_inds]
            batch_mol_merged = gather_mol(batch_mol)
            batch_props = self.prop_array[batch_inds]
            yield batch_mol_merged, batch_props



if __name__ == '__main__':
    train, val, test = load_fn()
    for i, batch in enumerate(train):
        print(batch)
        if i > 8:
            break


