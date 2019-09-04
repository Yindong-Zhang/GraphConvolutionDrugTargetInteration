import pandas as pd
import os
import tensorflow as tf
from src.featurizer import WeaveFeaturizer
from rdkit.Chem import MolFromSmiles
from src.data_utils import gather_mol
import numpy as np
tf.enable_eager_execution()

def create_dataset(batchsize = 32):
    dataDir = '../../data/kiba-origin/'
    df = pd.read_csv(os.path.join(dataDir, "kiba_props.csv"))
    smiles = df['smiles'].values
    props = df[[column for column in df.columns if column != 'smiles']].values
    dataset = tf.data.Dataset.from_tensor_slices((smiles, props))
    dataset.shuffle(2000).repeat().batch(batchsize)

    return dataset

class DataSet():
    def __init__(self, dataDir= '../../data/kiba-origin/', is_log= True, setting_no = 1):

        # self.charsmiset = CHARISOSMISET ###HERE CAN BE EDITED
        # self.charsmiset_size = CHARISOSMILEN
        self.filepath = dataDir
        self.problem_type = setting_no
        self.featurizer = WeaveFeaturizer()
        self.is_log = is_log
        df = pd.read_csv(os.path.join(dataDir, "kiba_props.csv"))
        self.smiles_list = df['smiles'].tolist()
        props = df[[column for column in df.columns if column != 'smiles']].values
        self.props = props
        self.rd_mols = [MolFromSmiles(smiles) for smiles in self.smiles_list]
        self.mol_list = self.featurizer(self.rd_mols)


    def iter_batch(self, batchsize, inds, shuffle = True, seed = 32):
        """

        :param batchsize:
        :param inds: a list of inds to use in dataset.
        :param shuffle:
        :param seed:
        :return:
        """
        mol_list = self.mol_list
        mol_array = np.array(mol_list)
        num_samples = len(mol_list)
        sample_inds = np.arange(num_samples)[inds]
        num_batches = len(sample_inds) // batchsize
        rng = np.random.RandomState(seed)
        if shuffle:
            rng.shuffle(sample_inds)

        for i in range(num_batches):
            batch_inds = sample_inds[ i * batchsize : ( i + 1 ) * batchsize]
            batch_mol = mol_array[batch_inds]
            batch_mol_merged = gather_mol(batch_mol)
            yield batch_mol_merged



if __name__ == '__main__':
    dataset = create_dataset(32)
    for batch in dataset:
        print(batch)
