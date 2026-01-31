import joblib
import os
from torch_scatter import scatter_mean, scatter_max
import torch


def load_protein():
    dir_path = "data/protein"
    subtype_name = ["ahr", "pxr", "car"]
    protein_dict = {}
    for sn in subtype_name:
        protein_path = os.path.join(dir_path, "%s_embedding.pkl" % sn)
        protein_load = joblib.load(protein_path).to("cpu")

        # idx = torch.zeros(len(protein_load)).long()
        # protein_dict[sn + "mean"] = scatter_mean(protein_load, idx)
        # protein_dict[sn + "max"] = scatter_max(protein_load, idx)
        protein_dict[sn] = protein_load
    return protein_dict