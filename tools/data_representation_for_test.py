from rdkit.Chem import (MolFromSmiles,
                        MolToInchiKey)
from tools.bit_choice import (bit_analysis,
                              get_fingerprint,
                              get_descriptors)
from tools.graph_calculate import SmilesUMA2Graph
import numpy as np
from unimol_tools import UniMolRepr
from tools.gemini.geminimol.model.GeminiMol import GeminiMol
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem



class Representation:
    def __init__(self, bit_choice=True, use_cache=True, full_smiles=None):
        self.finger_tools = None
        self.des_tools = None
        if bit_choice:
            if full_smiles is None:
                raise NotImplementedError
            self.finger_tools, self.des_tools = bit_analysis(full_smiles, use_cache=use_cache)
        self.uma_graph = SmilesUMA2Graph()

    def __call__(self, smiles):
        mol = MolFromSmiles(smiles)
        assert mol is not None
        fingerprint, descriptors = self.get_fingerprint_and_descriptors(mol)
        graph_with_h, features_h, graph_no_h, features_no_h = self.uma_graph(smiles)

        return descriptors, fingerprint, graph_with_h, features_h, graph_no_h, features_no_h

    def get_fingerprint_and_descriptors(self, mol):
        fingerprint = get_fingerprint(mol)
        descriptors, _ = get_descriptors([mol])
        if False:
            fingerprint = self.finger_tools.norm(fingerprint)
            descriptors = self.des_tools.norm(descriptors)
        return fingerprint, descriptors


def data_map_for_smiles(smiles):
    init_feature_tools = Representation(False, use_cache=True, full_smiles=None)
    inchi_key = MolToInchiKey(MolFromSmiles(smiles))
    descriptors, fingerprint, graph_with_h, features_h, graph_no_h, features_no_h = init_feature_tools(smiles)

    itm_dict = {
        "smiles": smiles,
        "inchi_key": inchi_key,
        "descriptors": descriptors,
        "fingerprint": fingerprint,
    }
    return itm_dict


def data_representation(smiles_list):
    valid_data_list = {}
    for idx, smi in enumerate(smiles_list):
        try:
            valid_data_list[idx] = data_map_for_smiles(smi)
        except Exception:
            print(f"Error: {smi}")

    # update unimol
    valid_data_list = calcaulte_unimol_features(valid_data_list)
    # update gemini
    valid_data_list = calculate_gemini_features(valid_data_list)
    return valid_data_list



def calculate_gemini_features(mapping_data):

    encoders = GeminiMol(
        "ulcyp",
        depth=0,
        custom_label=None,
        extrnal_label_list=['Cosine', 'Pearson', 'RMSE', 'Manhattan']
    )
    keys = list(mapping_data.keys())
    for k in tqdm(keys, total=len(keys)):
        smiles = mapping_data[k]["smiles"]

        unimol_features, graph = encoders.extract_features([smiles])

        mapping_data[k]["gemini"] = unimol_features[0]
    return mapping_data

def get_atom_coords(smiles_list):
    atom_info = []
    coords_info = []
    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol)
        atom_info.append(list(atom.GetAtomicNum() for atom in mol.GetAtoms()))
        coords_info.append(mol.GetConformers()[0].GetPositions())
    data_cover = {"atoms": atom_info, "coordinates": coords_info}
    return data_cover

def calcaulte_unimol_features(mapping_data):
    clf = UniMolRepr(data_type='molecule',
                     remove_hs=False,
                     model_name='unimolv2',  # avaliable: unimolv1, unimolv2
                     model_size='164m',  # work when model_name is unimolv2. avaliable: 84m, 164m, 310m, 570m, 1.1B.
                     device='cuda',
                     batch_size=8
                     )
    keys = list(mapping_data.keys())
    record_smiles = [mapping_data[k]["smiles"] for k in keys]


    record_smiles = record_smiles
    from rdkit.Chem import SDWriter
    all_mol = []
    for smi in record_smiles:
        all_mol.append(Chem.MolFromSmiles(smi))

    unimol_repr = clf.get_repr(record_smiles, return_atomic_reprs=True)

    unimol_features, unimol_atom = np.array(unimol_repr['cls_repr']), unimol_repr['atomic_reprs']
    unimol_coords, unimol_symbol = unimol_repr["atomic_coords"], unimol_repr["atomic_symbol"]


    if not isinstance(unimol_features, np.ndarray):
        unimol_features = np.array(unimol_features)

    for idx, key in enumerate(keys):
        mapping_data[key]["mol"] = all_mol[idx]
        mapping_data[key]["unimol"] = unimol_features[idx]
        mapping_data[key]["unimol_atom"] = unimol_atom[idx]
        mapping_data[key]["unimol_symbol"] = unimol_symbol[idx]

    return mapping_data