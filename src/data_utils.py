import numpy as np
import json
import pickle
import collections
from collections import OrderedDict
import os
from src.featurizer import WeaveFeaturizer
import random
from rdkit.Chem import MolFromSmiles
import math


## ######################## ##
#
#  Define CHARSET, CHARLEN
#
## ######################## ##

# CHARPROTSET = { 'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, \
#             'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, \
#             'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20, \
#             'O': 20, 'U': 20,.+
#             'B': (2, 11),
#             'Z': (3, 13),
#             'J': (7, 9) }
# CHARPROTLEN = 21

CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                "U": 19, "T": 20, "W": 21,
                "V": 22, "Y": 23, "X": 24,
                "Z": 25 }

PROTCHARSIZE = 25

CHARCANSMISET = { "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6,
                  ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12,
                  "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18,
                  "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24,
                  "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30,
                  "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36,
                  "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42,
                  "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48,
                  "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54,
                  "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
                  "t": 61, "y": 62}

CHARCANSMILEN = 62

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64


## ######################## ##
#
#  Encoding Helpers
#
## ######################## ##



#  Y = -(np.log10(Y/(math.pow(math.e,9))))

def one_hot_smiles(line, MAX_SMI_LEN, smi_ch_ind):
    X = np.zeros((MAX_SMI_LEN, len(smi_ch_ind))) #+1

    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i, (smi_ch_ind[ch]-1)] = 1

    return X #.tolist()

def one_hot_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
    X = np.zeros((MAX_SEQ_LEN, len(smi_ch_ind)))
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i, (smi_ch_ind[ch])-1] = 1

    return X #.tolist()


def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
    X = np.zeros(MAX_SMI_LEN)
    for i, ch in enumerate(line[:MAX_SMI_LEN]): #	x, smi_ch_ind, y
        X[i] = smi_ch_ind[ch]

    return X #.tolist()

def label_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
    X = np.zeros(MAX_SEQ_LEN)

    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]

    return X #.tolist()



## ######################## ##
#
#  DATASET Class
#
## ######################## ##

def load_data(filepath, seqlen, featurizer, problem_type, is_log):
    print("Read %s start" %(filepath, ))

    ligands = json.load(open(filepath + "ligands_can.json"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(filepath + "proteins.json"), object_pairs_hook=OrderedDict)

    Y = pickle.load(open(filepath + "Y", "rb"), encoding='latin1')  ### TODO: read from raw
    if is_log:
        Y = -(np.log10(Y / (math.pow(10, 9))))

    rd_mols = [MolFromSmiles(smiles) for smiles in ligands.values()]
    mol_list = featurizer(rd_mols)

    prot_list = [label_sequence(protein, seqlen, CHARPROTSET) for protein in proteins.values()]

    return mol_list, prot_list, Y

def load_5fold_split(filepath, setting_no):
    test_fold = json.load(open(filepath + "folds/test_fold_setting" + str(setting_no) + ".txt"))
    train_folds = json.load(open(filepath + "folds/train_fold_setting" + str(setting_no) + ".txt"))
    return train_folds, test_fold

def gather_mol(mols):
    """
    combine multiple moleculars in one big graph
    :param mols: batch of deepchem.weaveMol, consist of a tuple of (list of mol, list of label, list of weight)
    :return:batch of data consist of (atom features, pair_features, pair_inds, atom_inds, atom_to_pair), \
    in shape (num_atoms, atom_features), ( num_pairs, pair_features), (num_pairs, 1), (num_atoms, 1), (num_pairs, 2))
    in which num_pairs = sum of num_atoms * num_atoms in each molecular
    """
    atom_feat = []
    pair_feat = []
    atom_split = []
    atom_to_pair = []
    pair_split = []
    start = 0
    for im, mol in enumerate(mols):
        n_atoms = mol.get_num_atoms()
        # number of atoms in each molecule
        atom_split.extend([im] * n_atoms)
        # index of pair features
        C0, C1 = np.meshgrid(np.arange(n_atoms), np.arange(n_atoms))
        atom_to_pair.append(
            np.transpose(
                np.array([C1.flatten() + start,
                          C0.flatten() + start])))
        # number of pairs for each atom
        pair_split.extend(C1.flatten() + start)
        start = start + n_atoms

        # atom features
        atom_feat.append(mol.get_atom_features())
        # pair features
        pair_feat.append(
            np.reshape(mol.get_pair_features(),
                       (n_atoms * n_atoms, -1)))

    mol_merged = [
        np.concatenate(atom_feat, axis=0).astype(np.float32),
        np.concatenate(pair_feat, axis=0).astype(np.float32),
        np.array(pair_split),
        np.array(atom_split),
        np.concatenate(atom_to_pair, axis=0),
    ]
    return mol_merged

# works for large dataset
class DataSet():
    def __init__(self, fpath, seqlen, featurizer, is_log, setting_no = 1):
        self.SEQLEN = seqlen
        self.charseqset = CHARPROTSET
        # self.charseqset_size = CHARPROTLEN

        # self.charsmiset = CHARISOSMISET ###HERE CAN BE EDITED
        # self.charsmiset_size = CHARISOSMILEN
        self.filepath = fpath
        self.problem_type = setting_no
        self.featurizer = featurizer
        self.is_log = is_log
        self.mol_list, self.protSeq_list, self.labels = self.parse_data(is_log)

    def load_5fold_split(self): ### fpath should be the dataset folder /kiba/ or /davis/
        setting_no = self.problem_type
        print("Reading %s start" % self.filepath)

        test_fold = json.load(open(os.path.join(self.filepath, "folds/test_fold_setting" + str(setting_no)+".txt")))
        train_folds = json.load(open(os.path.join(self.filepath, "folds/train_fold_setting" + str(setting_no)+".txt")))

        return train_folds, test_fold

    def parse_data(self, is_log= True):
        print("Reading %s..." % self.filepath)

        ligands = json.load(open(os.path.join(self.filepath, "ligands_can.json")), object_pairs_hook=OrderedDict)
        proteins = json.load(open(os.path.join(self.filepath, "proteins.json")), object_pairs_hook=OrderedDict)

        Y = pickle.load(open(os.path.join(self.filepath, "Y"), "rb"), encoding='latin1') ### TODO: read from raw
        if is_log:
            Y = -(np.log10(Y/(math.pow(10,9))))

        rd_mols = [MolFromSmiles(smiles) for smiles in ligands.values()]
        mol_list = self.featurizer(rd_mols)

        prot_list = [label_sequence(protein, self.SEQLEN, self.charseqset) for protein in proteins.values()]

        return mol_list, prot_list, Y

    def iter_batch(self, batchsize, inds, shuffle = True, seed = 32):
        """

        :param batchsize:
        :param inds: a list of inds to use in dataset.
        :param shuffle:
        :param seed:
        :return:
        """
        mol_list, prot_list, aff_mat = self.mol_list, self.protSeq_list, self.labels
        mol_array = np.array(mol_list)
        prot_array = np.array(prot_list)
        mol_inds, prot_inds = np.where(np.isnan(aff_mat) == False)
        num_samples = mol_inds.shape[0]
        sample_inds = np.arange(num_samples)[inds]
        num_batches = len(sample_inds) // batchsize
        rng = np.random.RandomState(seed)
        if shuffle:
            rng.shuffle(sample_inds)

        for i in range(num_batches):
            batch_inds = sample_inds[ i * batchsize : ( i + 1 ) * batchsize]
            batch_mol_inds = mol_inds[batch_inds]
            batch_prot_inds = prot_inds[batch_inds]
            batch_mol = mol_array[batch_mol_inds]
            batch_prot = prot_array[batch_prot_inds]
            batch_mol_merged = gather_mol(batch_mol)
            labels = aff_mat[batch_mol_inds, batch_prot_inds].reshape(-1, 1)
            yield batch_mol_merged, batch_prot, labels

if __name__ == '__main__':
    print(projPath)
    # filepath = '../data/kiba/'
    # weave_featurizer = WeaveFeaturizer()
    # dataset = DataSet(fpath= filepath,  ### BUNU ARGS DA GUNCELLE
    #                   setting_no= 1,  ##BUNU ARGS A EKLE
    #                   seqlen= 1000,
    #                   featurizer= weave_featurizer,
    #                   is_log= True)
    # fold5_train, test_inds = dataset.load_5fold_split()
    # test_inds = np.array(test_inds)
    # for it in range(2):
    #     for i, t in enumerate(dataset.iter_batch(32, test_inds, shuffle= True, )):
    #         print(it, i)
    #         if i > 2:
    #             break
