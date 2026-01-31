from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from rdkit.Chem.AllChem import GetMorganGenerator
import numpy as np
import random
from collections import defaultdict

random.seed(42)
finger_gen = GetMorganGenerator(radius=2)


def smiles_to_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol


def get_murcko_scaffold(smiles):
    mol = smiles_to_mol(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)



def scaffold_split(smiles_list, labels, cutoff=0.7, train_ratio=0.8, seed=42, verbose=False):
    random.seed(seed)
    np.random.seed(seed)

    # ---------- Step 1: Calculate Murcko scaffolds ----------
    scaffolds = [get_murcko_scaffold(s) for s in smiles_list]
    unique_scaffolds, seen = [], set()
    for s in scaffolds:
        if s and s not in seen:
            seen.add(s)
            unique_scaffolds.append(s)

    # ---------- Step 2: Split by scaffolds without clustering ----------
    scaffold_to_indices = defaultdict(list)
    for i, scaffold in enumerate(scaffolds):
        scaffold_to_indices[scaffold].append(i)

    # ---------- Step 3: Label statistics ----------
    labels = np.array(labels)
    scaffold_stats = {}
    for scaffold, idxs in scaffold_to_indices.items():
        c_labels = labels[idxs]
        scaffold_stats[scaffold] = {
            "size": len(idxs),
            "pos": int((c_labels == 1).sum()),
            "neg": int((c_labels == 0).sum())
        }

    total_pos = int((labels == 1).sum())
    total_neg = int((labels == 0).sum())
    target_pos = int(total_pos * (1 - train_ratio))
    target_neg = int(total_neg * (1 - train_ratio))

    if verbose:
        print(f"[Label-balanced small-first split] Target test positives: {target_pos}, negatives: {target_neg}")

    # ---------- Step 4: small-first + dynamic ratio ----------
    scaffold_sorted = sorted(scaffold_stats.items(), key=lambda x: x[1]["size"])

    test_scaffolds, train_scaffolds = [], []
    test_pos, test_neg = 0, 0

    for scaffold, stat in scaffold_sorted:
        c_pos, c_neg = stat["pos"], stat["neg"]

        # Check if target is reached
        if test_pos >= target_pos and test_neg >= target_neg:
            train_scaffolds.append(scaffold)
            continue

        # Decide whether to add to test set based on class distribution
        add_to_test = False
        if test_pos < target_pos or test_neg < target_neg:
            add_to_test = True
            # Avoid over-adding any single class
            if (test_pos + c_pos > target_pos) and (c_pos > c_neg):
                add_to_test = False
            elif (test_neg + c_neg > target_neg) and (c_neg > c_pos):
                add_to_test = False

        if add_to_test:
            test_scaffolds.append(scaffold)
            test_pos += c_pos
            test_neg += c_neg
        else:
            train_scaffolds.append(scaffold)

    # ---------- Step 5: Defensive allocation ----------
    all_scaffolds = set(scaffold_to_indices.keys())
    unassigned = list(all_scaffolds - set(train_scaffolds) - set(test_scaffolds))
    if unassigned:
        train_scaffolds.extend(unassigned)  # All to training set, ensure no missing samples

    # ---------- Step 6: Aggregate indices ----------
    train_idx, test_idx = [], []
    for scaffold in train_scaffolds:
        train_idx.extend(scaffold_to_indices[scaffold])
    for scaffold in test_scaffolds:
        test_idx.extend(scaffold_to_indices[scaffold])

    assert len(set(train_idx).intersection(set(test_idx))) == 0, "overlap!"
    assert len(train_idx) + len(test_idx) == len(smiles_list), \
        f"missing samples! train+test={len(train_idx)+len(test_idx)} / total={len(smiles_list)}"

    random.shuffle(train_idx)
    random.shuffle(test_idx)

    if verbose:
        print(f"Total: {len(smiles_list)}")
        print(f"Train: {len(train_idx)} ({len(train_idx)/len(smiles_list):.2%})")
        print(f"Test : {len(test_idx)} ({len(test_idx)/len(smiles_list):.2%})")
        print(f"Pos_train={ (labels[train_idx]==1).sum() }, Pos_test={ (labels[test_idx]==1).sum() }")

    return train_idx, test_idx


def scaffold_split_with_fixed_test(smiles_list, labels, test_smiles=None,
                                   cutoff=0.7, train_ratio=0.8,
                                   seed=42, verbose=False, allow_scaffold_overlap=False):
    smiles_list = np.array(smiles_list)
    labels = np.array(labels)

    # Step 0: If test_smiles is empty, execute standard scaffold_split
    if not test_smiles:
        if verbose:
            print("[Info] No test_smiles provided, executing standard scaffold_split.")
        return scaffold_split(smiles_list, labels, cutoff, train_ratio, seed, verbose)

    # ---------- Step 1: Calculate scaffolds for test_smiles ----------
    def get_scaffold(s):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)

    # Calculate scaffolds for test_smiles
    test_scaffolds = [get_scaffold(s) for s in test_smiles]

    # ---------- Step 2: Extract compounds with same scaffolds as test_smiles ----------
    # Get indices of compounds in smiles_list with same scaffolds as test_smiles
    test_idx = []
    for i, s in enumerate(smiles_list):
        scaffold = get_scaffold(s)
        if scaffold in test_scaffolds:
            test_idx.append(i)

    # ---------- Step 3: Extract corresponding labels ----------
    test_labels = labels[test_idx].tolist()
    test_smiles_selected = smiles_list[test_idx].tolist()

    if verbose:
        print(f"[Scaffold extraction] Extracted {len(test_idx)} compounds with same scaffolds as test_smiles.")

    # ---------- Step 4: Extract remaining samples ----------
    remain_idx = [i for i in range(len(smiles_list)) if i not in test_idx]
    smiles_remain = smiles_list[remain_idx].tolist()
    labels_remain = labels[remain_idx].tolist()

    # ---------- Step 5: Execute standard scaffold_split on remaining samples ----------
    # Perform standard scaffold split on remaining samples to get train and test subset indices
    train_sub_idx, test_sub_idx = scaffold_split(
        smiles_remain, labels_remain, cutoff=cutoff, train_ratio=train_ratio,
        seed=seed, verbose=verbose
    )

    # ---------- Step 6: Map back to original indices ----------
    # Map train and test subset indices from standard split back to original smiles_list indices
    train_idx_final = [remain_idx[i] for i in train_sub_idx]
    test_idx_final = [remain_idx[i] for i in test_sub_idx]

    # ---------- Step 7: Merge test_smiles and standard split test sets ----------
    # Merge compounds with same scaffolds as test_smiles and those from standard split
    test_idx_final = sorted(set(test_idx + test_idx_final))

    # ---------- Step 8: Completeness check ----------
    # Ensure no overlap
    assert len(set(train_idx_final) & set(test_idx_final)) == 0, "Overlap detected!"
    assert len(train_idx_final) + len(test_idx_final) == len(smiles_list), \
        f"Total mismatch: train+test={len(train_idx_final) + len(test_idx_final)} != total={len(smiles_list)}"

    if verbose:
        print(f"[Final split result] Train: {len(train_idx_final)} ({len(train_idx_final) / len(smiles_list):.2%})")
        print(f"Test : {len(test_idx_final)} ({len(test_idx_final) / len(smiles_list):.2%})")

    return train_idx_final, test_idx_final