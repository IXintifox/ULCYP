import copy
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
from rdkit.Chem.AllChem import GetMorganGenerator

g_f6 = GetMorganGenerator(radius=3, includeChirality=True)
g_f4 =  GetMorganGenerator(radius=2, includeChirality=True)

def _scaffold_mol_from_smiles(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    scaff = MurckoScaffold.GetScaffoldForMol(m)
    if scaff is None or scaff.GetNumAtoms() == 0:
        return None
    return scaff

def _fp(mol, radius=2, nBits=2048):
    if mol is None:
        return None
    return g_f6.GetFingerprint(mol)

def label_check_again(mapping_data):
    all_smiles = []
    all_labels = []
    all_test_mask = []

    for itm in range(len(mapping_data)):
        all_smiles.append(mapping_data[itm]["smiles"])
        all_labels.append(mapping_data[itm]["label"])
        all_test_mask.append(mapping_data[itm]["test_label_mask"])

    all_smiles = np.array(all_smiles, dtype=object)
    all_labels = np.array(all_labels, dtype=object)
    all_test_mask = np.array(all_test_mask)

    test_all_mask_combine = np.sum(all_test_mask[:, 20:], axis=-1)
    test_smiles = all_smiles[test_all_mask_combine > 0]

    # Fill in: test set scaffolds and fingerprints
    test_scaffold = [_scaffold_mol_from_smiles(s) for s in test_smiles]
    test_scaffold_fingerprint = [fp for fp in (_fp(m) for m in test_scaffold) if fp is not None]

    new_mapping_data = {}

    for itm in range(len(mapping_data)):
        itm_mapping_data = mapping_data[itm]
        itm_smiles = itm_mapping_data["smiles"]

        # Fill in: sample scaffolds and fingerprints
        itm_scaffold = _scaffold_mol_from_smiles(itm_smiles)
        itm_scaffold_finger = _fp(itm_scaffold)

        # Calculate similarity with all test scaffolds
        itm_sim = []
        if itm_scaffold_finger is not None and len(test_scaffold_fingerprint) > 0:
            for test_f in test_scaffold_fingerprint:
                # Fill in: compare scaffold similarity
                itm_sim.append(DataStructs.TanimotoSimilarity(itm_scaffold_finger, test_f))

        new_test_mask = itm_mapping_data["test_label_mask"].copy()

        # Only move when similarity exists and max value > 0.7
        if itm_sim and max(itm_sim) > 0.7:
            # Original template had error finding keys from mapping_data, now changed to search within **sample**
            all_train_valid_keys = [k for k in itm_mapping_data.keys() if ("train" in k) or ("valid" in k)]

            for ky in all_train_valid_keys:
                origin_mask = itm_mapping_data[ky]
                new_mask = copy.deepcopy(origin_mask)

                # Avoid chained indexing: get tail first, then fill back
                tail = new_mask[20:].copy()
                idx = (origin_mask[20:] == 1)
                tail[idx] = 0
                new_mask[20:] = tail

                test_tail = new_test_mask[20:].copy()
                test_tail[idx] = 1
                new_test_mask[20:] = test_tail

                itm_mapping_data[ky] = new_mask

        itm_mapping_data["test_label_mask"] = new_test_mask
        itm_mapping_data.pop("fingerprint", None)
        new_mapping_data[itm] = itm_mapping_data
        itm_mapping_data["fingerprint4"] = g_f4.GetFingerprint(Chem.MolFromSmiles(itm_smiles))
        itm_mapping_data["fingerprint6"] = g_f6.GetFingerprint(Chem.MolFromSmiles(itm_smiles))
        new_mapping_data[itm] = itm_mapping_data

    return new_mapping_data