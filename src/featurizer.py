
import numpy as np
from rdkit import Chem

class Featurizer(object):
    """
    Abstract class for calculating a set of features for a molecule.

    Child classes implement the _featurize method for calculating features
    for a single molecule.
    """

    def featurize(self, mols, verbose=True, log_every_n=1000):
        """
        Calculate features for molecules.

        Parameters
        ----------
        mols : iterable
            RDKit Mol objects.
        """
        mols = list(mols)
        features = []
        for i, mol in enumerate(mols):
            if mol is not None:
                features.append(self._featurize(mol))
            else:
                features.append(np.array([]))

        features = np.asarray(features)
        return features

    def _featurize(self, mol):
        """
        Calculate features for a single molecule.

        Parameters
        ----------
        mol : RDKit Mol
            Molecule.
        """
        raise NotImplementedError('Featurizer is not defined.')

    def __call__(self, mols):
        """
        Calculate features for molecules.

        Parameters
        ----------
        mols : iterable
            RDKit Mol objects.
        """
        return self.featurize(mols)


def get_intervals(l):
    """For list of lists, gets the cumulative products of the lengths"""
    intervals = len(l) * [0]
    # Initalize with 1
    intervals[0] = 1
    for k in range(1, len(l)):
        intervals[k] = (len(l[k]) + 1) * intervals[k - 1]

    return intervals

def safe_index(l, e):
    """Gets the index of e in l, providing an index of len(l) if not found"""
    try:
        return l.index(e)
    except:
        return len(l)


possible_atom_list = [
    'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Mg', 'Na', 'Br', 'Fe', 'Ca', 'Cu',
    'Mc', 'Pd', 'Pb', 'K', 'I', 'Al', 'Ni', 'Mn', 'Unknown', # 事实上要不了这么多
]
possible_numH_list = [0, 1, 2, 3, 4]
possible_valence_list = [0, 1, 2, 3, 4, 5, 6]
possible_formal_charge_list = [-3, -2, -1, 0, 1, 2, 3]
possible_hybridization_list = [
    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2
]
possible_number_radical_e_list = [0, 1, 2]
possible_isAromatic_list = [0, 1] #atom.GetIsAromatic()
possible_chirality_list = ['R', 'S']
possible_degree_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # TODO: use degree features ?

atom_reference_lists = [
    possible_atom_list, possible_numH_list, possible_valence_list,
    possible_formal_charge_list, possible_number_radical_e_list,
    possible_hybridization_list, possible_isAromatic_list
]

intervals = get_intervals(atom_reference_lists)


def get_atom_feature_list(atom, onlyType = False):
    features = []
    features.append(safe_index(possible_atom_list, atom.GetSymbol()))
    if not onlyType:
        features.append(safe_index(possible_numH_list, atom.GetTotalNumHs()))
        features.append(safe_index(possible_valence_list, atom.GetImplicitValence()))
        features.append(safe_index(possible_formal_charge_list, atom.GetFormalCharge()))
        features.append(safe_index(possible_number_radical_e_list,
                                 atom.GetNumRadicalElectrons()))
        features.append(safe_index(possible_hybridization_list, atom.GetHybridization()))
        features.append(safe_index(possible_isAromatic_list, atom.GetIsAromatic()))
    return features


def features_to_id(features, intervals):
    """Convert list of features into index using spacings provided in intervals"""
    id = 0
    for k in range(len(intervals)):
        id += features[k] * intervals[k]

    # Allow 0 index to correspond to null molecule 1
    id = id + 1
    return id


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_to_id(atom):
    """Return a unique id corresponding to the atom type"""
    features = get_atom_feature_list(atom)
    return features_to_id(features, intervals)


def atom_features_onehot(atom,
                  bool_id_feat=False,
                  explicit_H=False,
                  use_chirality=False):
    if bool_id_feat:
        return np.array([atom_to_id(atom)])
    else:
        from rdkit import Chem
        results = one_of_k_encoding_unk(
            atom.GetSymbol(),
            ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Mg', 'Na', 'Br', 'Fe', 'Ca', 'Cu',
                    'Mc', 'Pd', 'Pb', 'K', 'I', 'Al', 'Ni', 'Mn', 'Unknown'
            ]) + one_of_k_encoding(atom.GetDegree(),
                                   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
                  one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  one_of_k_encoding_unk(atom.GetHybridization(), [
                      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2
                  ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if not explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                      [0, 1, 2, 3, 4])
        if use_chirality:
            try:
                results = results + one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            except:
                results = results + [False, False
                                     ] + [atom.HasProp('_ChiralityPossible')]

        return np.array(results)

def bond_features_onehot(bond, use_chirality=False):
    from rdkit import Chem
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats)

possible_bondType_list = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
possible_conjuate_list = [0, 1]
possible_inRing_list = [0, 1]
bond_reference_lists = [possible_bondType_list, possible_conjuate_list, possible_inRing_list]

def get_bond_feature_list(bond, use_chirality=False):
    feat = [0] * len(bond_reference_lists)
    feat[0] = safe_index(possible_bondType_list, bond.GetBondType())
    feat[1] = safe_index(possible_conjuate_list, bond.GetIsConjugated())
    feat[2] = safe_index(possible_inRing_list, bond.IsInRing())
    return feat


def pair_features(mol, edge_list, canon_adj_list, bt_len=6,
                  graph_distance=True):
    if graph_distance:
        max_distance = 7
    else:
        max_distance = 1
    N = mol.GetNumAtoms()
    features = np.zeros((N, N, bt_len + max_distance + 1))
    num_atoms = mol.GetNumAtoms()
    rings = mol.GetRingInfo().AtomRings()
    for a1 in range(num_atoms):
        for a2 in canon_adj_list[a1]:
            # first `bt_len` features are bond features(if applicable)
            features[a1, a2, :bt_len] = np.asarray(
                edge_list[tuple(sorted((a1, a2)))], dtype=float)
        for ring in rings: # 增加特征： 是否处于同一环中
            if a1 in ring:
                # `bt_len`-th feature is if the pair of atoms are in the same ring
                features[a1, ring, bt_len] = 1
                features[a1, a1, bt_len] = 0.
        # graph distance between two atoms
        if graph_distance: # 增加特征， 距离在max-distance 的原子，通过在邻接矩阵对应位置置为1表示
            distance = find_distance(
                a1, num_atoms, canon_adj_list, max_distance=max_distance)
            features[a1, :, bt_len + 1:] = distance
    # Euclidean distance between atoms
    if not graph_distance:
        coords = np.zeros((N, 3))
        for atom in range(N):
            pos = mol.GetConformer(0).GetAtomPosition(atom)
            coords[atom, :] = [pos.x, pos.y, pos.z]
        features[:, :, -1] = np.sqrt(np.sum(np.square(
            np.stack([coords] * N, axis=1) - \
            np.stack([coords] * N, axis=0)), axis=2))

    return features

def find_distance(a1, num_atoms, canon_adj_list, max_distance=7):
    distance = np.zeros((num_atoms, max_distance))
    radial = 0
    # atoms `radial` bonds away from `a1`
    adj_list = set(canon_adj_list[a1])
    # atoms less than `radial` bonds away
    all_list = set([a1])
    while radial < max_distance:
        distance[list(adj_list), radial] = 1
        all_list.update(adj_list)
        # find atoms `radial`+1 bonds away
        next_adj = set()
        for adj in adj_list:
            next_adj.update(canon_adj_list[adj])
        adj_list = next_adj - all_list
        radial = radial + 1
    return distance


class WeaveFeaturizer(Featurizer):
    name = ['weave_mol']

    def __init__(self, only_atom_type = False,  graph_distance=False, explicit_H=False,
                 use_chirality=False):
        # Distance is either graph distance(True) or Euclidean distance(False,
        # only support datasets providing Cartesian coordinates)
        self.only_atom_type = only_atom_type

        self.graph_distance = graph_distance
        # Set dtype
        self.dtype = object
        # If includes explicit hydrogens
        self.explicit_H = explicit_H
        # If uses use_chirality
        self.use_chirality = use_chirality

    def _featurize(self, mol):
        """Encodes mol as a WeaveMol object."""
        # Atom features
        idx_nodes = [(a.GetIdx(),
                      get_atom_feature_list(a, onlyType= self.only_atom_type))
                     for a in mol.GetAtoms()]

        idx_nodes.sort()  # Sort by ind to ensure same order as rd_kit
        idx, nodes = list(zip(*idx_nodes))

        # Stack nodes into an array
        atom_input = np.array(nodes)
        # atom_input = [nodes[:, i] for i in range(nodes.shape[1])] # split array column wise

        # Get bond lists
        edge_feat_ls = []
        for b in mol.GetBonds():
            edge_feat_ls.append(
                ((b.GetBeginAtomIdx(), b.GetEndAtomIdx()), get_bond_feature_list(b, use_chirality=self.use_chirality))
                )
            edge_feat_ls.append(
                ((b.GetEndAtomIdx(), b.GetBeginAtomIdx()), get_bond_feature_list(b, use_chirality=self.use_chirality))
            )

        edge_feat_ls.sort(key= lambda edge: edge[0])

        # Get canonical adjacency list
        # canon_adj_list = [[] for mol_id in range(len(nodes))]
        # for edge in edge_feat_ls.keys():
        #     canon_adj_list[edge[0]].append(edge[1])
        #     canon_adj_list[edge[1]].append(edge[0])

        atom2pair_ls = []
        pair_ls = []
        for edge_feat in edge_feat_ls:
            atom2pair_ls.append(edge_feat[0])
            pair_ls.append(edge_feat[1])

        atom2pair= np.array(atom2pair_ls)
        pair_input = np.array(pair_ls)
        # pair_input = [pairs[:, i] for i in range(pairs.shape[1])]

        # Calculate pair features
        # pairs = pair_features(
        #     mol,
        #     edge_feat_dict,
        #     canon_adj_list,
        #     bt_len=6,
        #     graph_distance=self.graph_distance)

        return (atom_input, pair_input, atom2pair)

    @property
    def atom_cat_dim(self):
        if self.only_atom_type:
            return [len(atom_reference_lists[0]) + 1]
        else:
            return [len(feat) + 1 for feat in atom_reference_lists]

    @property
    def bond_cat_dim(self):
        return [len(feat) + 1 for feat in bond_reference_lists]