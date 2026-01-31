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


def butina_cluster(fps, cutoff=0.7):
    dists = []
    nfps = len(fps)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])
    clusters = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
    return clusters


def scaffold_split(smiles_list, labels, cutoff=0.7, train_ratio=0.8, seed=42, verbose=False):
    random.seed(seed)
    np.random.seed(seed)

    scaffolds = [get_murcko_scaffold(s) for s in smiles_list]
    unique_scaffolds, seen = [], set()
    for s in scaffolds:
        if s and s not in seen:
            seen.add(s)
            unique_scaffolds.append(s)

    scaffold_mols = [Chem.MolFromSmiles(s) for s in unique_scaffolds if Chem.MolFromSmiles(s)]
    fps = [Chem.RDKFingerprint(m) for m in scaffold_mols]
    clusters = butina_cluster(fps, cutoff=cutoff)
    clusters = [tuple(sorted(c)) for c in clusters]
    clusters = sorted(clusters, key=lambda x: len(x), reverse=True)

    scaffold_to_cluster = {}
    for c_id, cluster in enumerate(clusters):
        for idx in cluster:
            scaffold_to_cluster[unique_scaffolds[idx]] = c_id

    mol_cluster_ids = [scaffold_to_cluster.get(s, -1) for s in scaffolds]
    cluster_to_indices = defaultdict(list)
    for i, c_id in enumerate(mol_cluster_ids):
        if c_id != -1:
            cluster_to_indices[c_id].append(i)
        else:
            cluster_to_indices[f"single_{i}"].append(i)

    labels = np.array(labels)
    cluster_stats = {}
    for cid, idxs in cluster_to_indices.items():
        c_labels = labels[idxs]
        cluster_stats[cid] = {
            "size": len(idxs),
            "pos": int((c_labels == 1).sum()),
            "neg": int((c_labels == 0).sum())
        }

    total_pos = int((labels == 1).sum())
    total_neg = int((labels == 0).sum())
    target_pos = int(total_pos * (1 - train_ratio))
    target_neg = int(total_neg * (1 - train_ratio))

    if verbose:
        print(f"[Label-balanced small-first split]")
        print(f"Target test positives: {target_pos}, negatives: {target_neg}")

    clusters_sorted = sorted(cluster_stats.items(), key=lambda x: x[1]["size"])

    test_clusters, train_clusters = [], []
    test_pos, test_neg = 0, 0

    for cid, stat in clusters_sorted:
        c_pos, c_neg = stat["pos"], stat["neg"]

        if test_pos >= target_pos and test_neg >= target_neg:
            train_clusters.append(cid)
            continue

        add_to_test = False
        if test_pos < target_pos or test_neg < target_neg:
            add_to_test = True
            if (test_pos + c_pos > target_pos) and (c_pos > c_neg):
                add_to_test = False
            elif (test_neg + c_neg > target_neg) and (c_neg > c_pos):
                add_to_test = False

        if add_to_test:
            test_clusters.append(cid)
            test_pos += c_pos
            test_neg += c_neg
        else:
            train_clusters.append(cid)

    all_cids = set(cluster_to_indices.keys())
    unassigned = list(all_cids - set(train_clusters) - set(test_clusters))
    if unassigned:
        train_clusters.extend(unassigned)

    train_idx, test_idx = [], []
    for cid in train_clusters:
        train_idx.extend(cluster_to_indices[cid])
    for cid in test_clusters:
        test_idx.extend(cluster_to_indices[cid])

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

    if not test_smiles:
        if verbose:
            print("[Info] No test_smiles provided, executing standard scaffold_split.")
        return scaffold_split(smiles_list, labels, cutoff, train_ratio, seed, verbose)

    def get_inchikey(s):
        mol = Chem.MolFromSmiles(s)
        return Chem.MolToInchiKey(mol) if mol else None

    all_inchikeys = [get_inchikey(s) for s in smiles_list]
    test_inchikeys = {get_inchikey(s) for s in test_smiles if get_inchikey(s)}

    found_idx = [i for i, key in enumerate(all_inchikeys) if key in test_inchikeys]
    found_test = [smiles_list[i] for i in found_idx]
    missing_test = [s for s in test_smiles if get_inchikey(s) not in set(all_inchikeys)]

    if len(found_idx) == 0:
        print("Warning: No test_smiles found in smiles_list, executing standard scaffold_split.")
        return scaffold_split(smiles_list, labels, cutoff, train_ratio, seed, verbose)

    if len(missing_test) > 0:
        print(f"Warning: {len(missing_test)} test_smiles not found in dataset, ignored.")

    if verbose:
        print(f"[Reserved test set identification] Matched {len(found_idx)} / {len(test_smiles)} test_smiles.")

    remain_idx = [i for i in range(len(smiles_list)) if i not in found_idx]
    smiles_remain = smiles_list[remain_idx].tolist()
    labels_remain = labels[remain_idx].tolist()

    def get_scaffold(s):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)

    test_scaffolds = {get_scaffold(s) for s in found_test if get_scaffold(s)}
    remain_scaffolds = {get_scaffold(s) for s in smiles_remain if get_scaffold(s)}
    overlap = test_scaffolds & remain_scaffolds

    if verbose:
        print(f"[Scaffold overlap detection] Test scaffolds: {len(test_scaffolds)}, Train candidate scaffolds: {len(remain_scaffolds)}")
        print(f"-> Overlap scaffold count: {len(overlap)}")

    if len(overlap) > 0:
        msg = f"Warning: Detected {len(overlap)} scaffolds in both test_smiles and training candidates."
        if allow_scaffold_overlap:
            if verbose: print(msg + "Continuing with scaffold_split.")
        else:
            raise ValueError("❌ " + msg + " Set allow_scaffold_overlap=False, execution terminated.")

    test_scaffolded_idx = [i for i, s in enumerate(smiles_remain) if get_scaffold(s) in test_scaffolds]
    test_idx_scaffolded = [remain_idx[i] for i in test_scaffolded_idx]
    test_idx = sorted(set(found_idx + test_idx_scaffolded))

    remain_sub_idx = [i for i in range(len(smiles_remain)) if i not in test_scaffolded_idx]
    train_sub_idx, _ = scaffold_split(
        [smiles_remain[i] for i in remain_sub_idx],
        [labels_remain[i] for i in remain_sub_idx],
        cutoff=cutoff, train_ratio=train_ratio,
        seed=seed, verbose=verbose
    )

    train_idx = [remain_idx[i] for i in train_sub_idx]
    train_idx = sorted(set(train_idx) - set(test_idx))

    assert len(set(train_idx) & set(test_idx)) == 0, "Overlap detected!"
    assert len(train_idx) + len(test_idx) == len(smiles_list), \
        f"Total mismatch: train+test={len(train_idx) + len(test_idx)} != total={len(smiles_list)}"

    if verbose:
        print(f"[Final split result] Train: {len(train_idx)} ({len(train_idx) / len(smiles_list):.2%})")
        print(f"Test : {len(test_idx)} ({len(test_idx) / len(smiles_list):.2%})")

    return train_idx, test_idx