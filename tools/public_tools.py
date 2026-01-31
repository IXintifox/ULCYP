from easydict import EasyDict
import yaml
import numpy as np
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from rdkit.Chem.AllChem import GetMorganGenerator
from rdkit.ML.Cluster import Butina
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
from collections import Counter
fp = GetMorganGenerator()

def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))

def compute_morgan_fingerprint(smiles, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return fp.GetFingerprint(mol).ToBitString()
    return None

def get_butina_cluster(mol_list):
    fps = [fp.GetCountFingerprint(m) for m in mol_list]

    dists = []
    for i in tqdm(range(1, len(fps)), total=len(fps), disable=True):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])

    cutoff = 0.3
    # cutoff = 0.1
    clusters = Butina.ClusterData(dists, len(fps), cutoff, isDistData=True)

    idx2cluster = {}
    for i,clus in enumerate(clusters):
        for itm in clus:
            idx2cluster[itm] = i
    cluster_idx = []
    for itm in range(len(mol_list)):
        cluster_idx.append(idx2cluster[itm])
    return cluster_idx, clusters


def select_representatives_grouped(cluster_labels, group_size=300):
    cluster_counts = dict(Counter(cluster_labels))
    sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
    representatives = []
    for label, _ in sorted_clusters:
        members = np.where(np.array(cluster_labels) == label)[0]
        if len(members) > 0:
            representatives.append(members[0])

    representatives_grouped = {}
    total_clusters = len(sorted_clusters)
    num_groups = (total_clusters + group_size - 1) // group_size

    for i in range(num_groups):
        start = i * group_size
        end = (i + 1) * group_size
        representatives_grouped[i] = representatives[start:end]

    return representatives_grouped


def select_representatives_no_label(cluster_labels, total_representatives=300):
    representatives_above = []

    cluster_label_with_number = dict(Counter(cluster_labels))
    sorted_clusters = sorted(cluster_label_with_number.items(), key=lambda x: x[1], reverse=True)
    for label in np.unique(cluster_labels):
        members = np.where(cluster_labels == label)[0]
        if len(members) > 0:
            representatives_above.append(members[0])

    print("representatives_above: ", len(representatives_above))
    total_above = min(len(representatives_above), total_representatives)

    extra_needed = total_representatives
    if extra_needed > 0:
        representatives = np.array(representatives_above[:total_above])
    else:
        representatives = np.array(representatives_above)
    return representatives


def scaffold_split(smiles_idx, smiles):
    # smiles_idx = smiles_idx[:4000]
    # smiles = smiles[:4000]
    smiles_csv = pd.DataFrame(np.array([smiles_idx, smiles]).T, columns=["idx", "SMILES"])
    smiles_csv['Mol'] = smiles_csv['SMILES'].apply(Chem.MolFromSmiles)
    smiles_csv['fingerprint'] = smiles_csv['Mol'].apply(
        lambda mol: compute_morgan_fingerprint(Chem.MolToSmiles(mol)) if mol else None)

    smiles_csv["scaffold"] = smiles_csv["Mol"].apply(
        MurckoScaffold.GetScaffoldForMol
    )
    clusters_map, cluster = get_butina_cluster(smiles_csv["scaffold"].values.tolist())
    return cluster


def scaffold_split_no_label(smiles_idx, smiles):
    smiles_csv = pd.DataFrame(np.array([smiles_idx, smiles]).T, columns=["idx", "SMILES"])
    smiles_csv['Mol'] = smiles_csv['SMILES'].apply(Chem.MolFromSmiles)
    smiles_csv['fingerprint'] = smiles_csv['Mol'].apply(
        lambda mol: compute_morgan_fingerprint(Chem.MolToSmiles(mol)) if mol else None)

    smiles_csv["scaffold"] = smiles_csv["Mol"].apply(
        MurckoScaffold.GetScaffoldForMol
    )
    clusters = get_butina_cluster(smiles_csv["scaffold"].values.tolist())

    selected_indices = select_representatives_grouped(clusters, 400)
    selected_smiles = {}
    for i in list(selected_indices.keys()):
        selected_smiles[i] = smiles_idx[selected_indices[i]]
    return selected_smiles