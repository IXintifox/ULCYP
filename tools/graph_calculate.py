from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from tools.graph_tools import MolecularEncoder
import copy

def seq_len(seq):
    seq_dc = copy.deepcopy(seq)
    seq_dc = "".join(seq_dc)
    seq_dc = seq_dc.replace("Cl", "A")
    seq_dc = seq_dc.replace("Br", "D")
    seq_dc = seq_dc.replace("As", "G")
    return len(seq_dc), seq_dc

class SmilesUMA2Graph:
    def __init__(self):
        self.graph_tools_with_h = MolecularEncoder(h=True)
        self.graph_tools = MolecularEncoder()

    def __call__(self, smi):
        get_3d_mol = self.convert3d(smi)
        if get_3d_mol == 0:
            return (0,0,0,0)

        total_charge = sum([atom.GetFormalCharge() for atom in get_3d_mol.GetAtoms()])
        updated_mol_remove_h = Chem.RemoveHs(get_3d_mol)
        features_h, bonds_features_h, connect_h, n_atoms_h = self.graph_tools_with_h(get_3d_mol)
        features, bonds_features, connect, n_atoms = self.graph_tools(updated_mol_remove_h)
        return (
        (features_h, bonds_features_h, connect_h, n_atoms_h),
            features_h,
        (features, bonds_features, connect, n_atoms),
            features
        )

    @staticmethod
    def features(mol):
        atoms = mol.GetAtoms()
        record_features = []
        for i in range(len(atoms)):
            atom = mol.GetAtomWithIdx(i)
            get_atom_feature = np.array(eval(atom.GetProp("uma_feature")))
            record_features.append(get_atom_feature)
        return np.array(record_features)

    @staticmethod
    def convert3d(smi):
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        return_true = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if return_true != 0:
            return_true = AllChem.Compute2DCoords(mol)
            if return_true != 0:
                return 0
        AllChem.MMFFOptimizeMolecule(mol)
        return mol

    def extract_embedding(self, seq, position, charge=0):
        self.uma_predict.fit(seq, position, mol_charge=charge)
        features = self.uma_predict.embedding
        return seq, features

    def mapping_mol(self, mol, seq, features):
        seq_l, seq_rep = seq_len(seq)
        assert seq_l == len(features), "Not matched :%s %s" % (seq_l, len(features))
        atoms = mol.GetAtoms()
        for i in range(len(atoms)):
            sym = atoms[i].GetSymbol()
            if sym == "Br":
                sym = "D"
            elif sym == "Cl":
                sym = "A"
            elif sym == "As":
                sym = "G"
            assert seq_rep[i] == sym, "Not matched"
            atom = mol.GetAtomWithIdx(i)
            atom.SetProp("uma_feature", str(features[i].tolist()))

        for i in range(len(atoms)):
            atom = mol.GetAtomWithIdx(i)
            get_atom_feature = np.array(eval(atom.GetProp("uma_feature")))
            assert (features[i] == get_atom_feature).all(), "Feature mismatch"

        return mol


    @staticmethod
    def seq_position_map(mol):
        atoms = mol.GetAtoms()
        seq = []
        positions = []
        molecule_position = mol.GetConformer().GetPositions()
        for atx, atom in enumerate(atoms):
            seq.append(atom.GetSymbol())
            positions.append(molecule_position[atx].tolist())

        seq = ''.join(seq)
        return seq, positions


if __name__ == "__main__":
    tools = SmilesUMA2Graph()
    tools("CCc1cc(Cl)c(OC)c(C(=O)NCC2CCCN2CC)c1O")
