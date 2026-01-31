import os
# from graph_gen import gen_plot
# from dgllife.utils import CanonicalAtomFeaturizer
from rdkit import Chem
from rdkit.Chem import SanitizeMol
import pandas as pd
import numpy as np
from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.data.in_memory_dataset import InMemoryDataset
import torch
import copy
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType
from networkx import Graph
from tqdm import tqdm

max_p_length = 42


def _map(*args):
    return list(map(args[0], args[1]))

class MolecularEncoder:
    """
    encoder_unknown: True False or GCNN_paper
    encoder_vector: DGL or GCNN_paper
    """

    # get feature vectors
    def __init__(self, encoder_unknown=True, encoder_vector=None, h=False):
        self.encoder_unknown = encoder_unknown
        # symbol
        if h:
            self.atom_list = ['H', 'C', 'O', 'N', 'P', 'S', 'Cl', 'Br', 'I', 'F']
        else:
            self.atom_list = ['C', 'O', 'N', 'P', 'S', 'Cl', 'Br', 'I', 'F']


        # degree
        self.degree = list(range(6))

        # explicit valence
        self.explicit_valence = list(range(7))

        self.formal_charge_range = list(range(-1, 2))

        # hybridization
        self.hybridization = [HybridizationType.SP,
                              HybridizationType.SP2,
                              HybridizationType.SP3,
                              HybridizationType.SP3D,
                              HybridizationType.SP3D2
                              ]

        self.total_num_H = list(range(5))

        # =========================> bond_info
        self.bond_type = [BondType.SINGLE,
                          BondType.DOUBLE,
                          BondType.TRIPLE,
                          BondType.AROMATIC]


        self.bond_chirality = [
            "STEREONONE",
            "STEREOANY",
            "STEREOZ",
            "STEREOE"
        ]


    def __call__(self, mol):
        n_atom, x, b = self.process_mol_info(mol)
        edges_connect = self.get_bond(mol)
        return np.array(x, dtype=int), np.array(np.concatenate([b, b], axis=0), dtype=int), edges_connect.T, n_atom

    @staticmethod
    def one_hot_encoder(atom, match_set, encoder_unknown=False):
        temp_list = copy.deepcopy(match_set)
        if encoder_unknown:
            temp_list.append("<UNK>")
            if atom in temp_list:
                return [atom == a for a in temp_list]
            else:
                atom = "<UNK>"
                return [atom == a for a in temp_list]
        # other encoder
        else:
            return [atom == a for a in temp_list]

    def get_chirality(self, atom, encoder_unknown=False):
        c_pos = atom.HasProp('_ChiralityPossible')
        try:
            atom_c = self.get_mol_info(atom.GetProp('_CIPCode'), ['R', 'S'], encoder_unknown=encoder_unknown)
        except KeyError:
            if encoder_unknown:
                atom_c = [0, 0, 0]
            else:
                atom_c = [0, 0]
        atom_c = list(atom_c)
        atom_c.append(c_pos)
        return np.array(atom_c)

    def process_mol_info(self, mol):
        full_a_features = list()
        full_b_features = list()
        atom_i = defaultdict(list)
        bond_i = defaultdict(list)
        get_atom = mol.GetAtoms()
        get_bond = mol.GetBonds()
        atom_i['Symbol'] = _map(lambda x: self.get_mol_info(x.GetSymbol(),
                                                             self.atom_list,
                                                             encoder_unknown=self.encoder_unknown), get_atom)
        atom_i['Degree'] = _map(lambda x: self.get_mol_info(x.GetDegree(),
                                                             self.degree,
                                                             encoder_unknown=False), get_atom)
        atom_i['FormalCharge'] = _map(lambda x: self.get_mol_info(x.GetFormalCharge(),
                                                                   self.formal_charge_range,
                                                                   encoder_unknown=self.encoder_unknown), get_atom)
        atom_i['Hybridization'] = _map(lambda x: self.get_mol_info(x.GetHybridization(),
                                                                    self.hybridization,
                                                                    encoder_unknown=self.encoder_unknown), get_atom)
        atom_i['Total_num_H'] = _map(lambda x: self.get_mol_info(x.GetTotalNumHs(),
                                                                  self.total_num_H,
                                                                  encoder_unknown=self.encoder_unknown), get_atom)
        atom_i['Aromatic'] = _map(lambda x: np.array([x.GetIsAromatic()]), get_atom)

        # if mk:
        #     # DGL
        #     atom_i['Radical_electrons'] = _map(lambda x: self.get_mol_info(x.GetNumRadicalElectrons(),
        #                                                                     self.dgl_radical_electrons_range,
        #                                                                     encoder_unknown=self.encoder_unknown),
        #                                        get_atom)
        #     atom_i['Chirality'] = _map(lambda x: self.get_chirality(x, encoder_unknown=self.encoder_unknown),
        #                                get_atom)
        # else:
            # atom_i['Radical_electrons'] = _map(lambda x: self.get_mol_info(x.GetNumRadicalElectrons(),
            #                                                                 self.dgl_radical_electrons_range,
            #                                                                 encoder_unknown=self.encoder_unknown),
            #                                    get_atom)
        atom_i['Chirality'] = _map(lambda x: self.get_chirality(x, encoder_unknown=self.encoder_unknown),
                                   get_atom)

        bond_i["Bond_type"] = _map(lambda x: self.get_mol_info(x.GetBondType(),
                                                                  self.bond_type,
                                                                  encoder_unknown=self.encoder_unknown), get_bond)
        # bond_i["Stereo"] = _map(lambda x: self.get_mol_info(x.GetStereo(),
        #                                                        self.bond_chirality,
        #                                                        encoder_unknown=self.encoder_unknown), get_bond)
        bond_i["Is_in_ring"] = _map(lambda x: np.array([x.IsInRing()]), get_bond)
        # bond_i["Conjugate"] = _map(lambda x: np.array([x.GetIsConjugated()]), get_bond)
        atom_f_keys = list(atom_i.keys())
        bond_f_keys = list(bond_i.keys())

        for k in atom_f_keys:
            full_a_features += [atom_i[k]]
        full_a_features = np.concatenate(full_a_features, axis=1)
        record_atom_features, record_bond_features = [], []
        for i in full_a_features:
            record_atom_features += [i]

        for b in bond_f_keys:
            full_b_features += [bond_i[b]]
        full_b_features = np.concatenate(full_b_features, axis=1)
        for j in full_b_features:
            record_bond_features += [j]

        return mol.GetNumAtoms(), record_atom_features, record_bond_features

    def get_mol_info(self, atom, match_set, encoder_unknown=None):
        return np.array(self.one_hot_encoder(atom, match_set=match_set, encoder_unknown=encoder_unknown))

    @staticmethod
    def get_bond(mol):
        bonds_info = mol.GetBonds()
        bond_s_1 = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in bonds_info]
        bond_s_2 = [(b.GetEndAtomIdx(), b.GetBeginAtomIdx()) for b in bonds_info]
        get_b = np.concatenate([bond_s_1, bond_s_2], axis=0).T
        # G = Graph(bond_s).to_directed()
        # get_b = np.array(G.edges).T
        return get_b

if __name__ == '__main__':
    smiles = "C[C@H](O)c1ccccc1S(=O)(=O)c1ccc(/C=C/c2ccc(F)cc2)cc1"
    gen_plot = MolecularEncoder()
    features, bonds_features, connect, n_atoms = gen_plot(Chem.MolFromSmiles(smiles))
