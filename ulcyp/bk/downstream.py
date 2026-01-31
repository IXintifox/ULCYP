import torch.nn as nn
import torch
from easydict import EasyDict
import torch.nn.functional as F
import yaml
import numpy as np
from torch_geometric.utils import to_dense_batch
from torch.nn.modules import ModuleDict
from torch_scatter import (scatter_max,
                           scatter_mean)


class TaskClass(object):
    def __init__(self, sub_model_num):
        self.pu_task_idx = np.array(list(range(4)))
        self.loop_special_task_idx = {i: self.pu_task_idx + i * len(self.pu_task_idx) for i in range(0, sub_model_num)}


class DownStreamTask(nn.Module):
    def __init__(self, protein_input_dim, protein_hidden_dim, hidden_dim, head_n=16, layer_num=3, device="cpu", sub_model_num=5):
        super(DownStreamTask, self).__init__()
        self.sub_model_num = sub_model_num

        self.inducer_model_seq = torch.nn.ModuleDict()
        for model_num in list(range(sub_model_num)):
            self.inducer_model_seq["inducer_model_" + str(model_num)] = InducerBlock(protein_input_dim, protein_hidden_dim,hidden_dim, head_n=head_n,
                                          layer_num=layer_num,device=device)


    def forward(self, atom, protein, batch, protein_mode="mean"):
        record_all_sub_inducer, record_all_sub_inducer_features = [], []

        for model_idx in list(range(self.sub_model_num)):

            inducer_vector, inducer_molecule_features =\
                self.inducer_model_seq["inducer_model_" + str(model_idx)](atom, protein, batch)
            record_all_sub_inducer.append(inducer_vector)
            record_all_sub_inducer_features.append(inducer_molecule_features)

        record_all_sub_inducer = torch.cat(record_all_sub_inducer, dim=-1)
        record_all_sub_inducer_features = torch.cat(record_all_sub_inducer_features, dim=1)


        return  record_all_sub_inducer, record_all_sub_inducer_features,  None


class InducerBlock(nn.Module):
    def __init__(self, protein_input_dim, protein_hidden_dim, output_dim, head_n, layer_num, device="cpu"):
        super(InducerBlock, self).__init__()
        self.pu_task = PUTask(protein_input_dim, protein_hidden_dim, output_dim, head_n, layer_num, device="cpu")

    def forward(self, atom, protein, batch):
        low_out_share, low_output_layer, norm_low_attention = self.pu_task(atom, protein, batch)
        return low_output_layer, low_out_share

class BaseTask(nn.Module):
    def __init__(self, protein_input_dim, protein_hidden_dim, output_hidden_dim,  head_n, layer_num, device= "cpu"):
        super(BaseTask, self).__init__()
        dropout = 0.1
        self.output_projection = nn.Sequential(
            nn.Linear(256, output_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_hidden_dim, output_hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_hidden_dim//2, 1)
        )

        self.gate_attention = GatedMeanPooling()
        self.protein_attention_ahr = ProteinAttentionPooling()
        self.protein_attention_pxr = ProteinAttentionPooling()
        self.protein_attention_car = ProteinAttentionPooling()

class PUTask(BaseTask):
    def __init__(self, protein_input_dim, protein_hidden_dim, output_dim, head_n, layer_num, device):
        super(PUTask, self).__init__(protein_input_dim, protein_hidden_dim, output_dim, head_n, layer_num, device)
        self.task_num = 4
        self.cls = nn.Parameter(torch.zeros(1, 1, 256))
        self.single_net = SFilM()
        self.multi_net1 = MFiLM()
        self.multi_net2 = MFiLM()
        self.multi_net3 = MFiLM()

    def forward(self, atom, protein, batch):
        ahr_protein = protein["ahr"].to(atom)
        pxr_protein = protein["pxr"].to(atom)
        car_protein = protein["car"].to(atom)

        ahr_protein = self.protein_attention_ahr(ahr_protein)
        pxr_protein = self.protein_attention_pxr(pxr_protein)
        car_protein = self.protein_attention_car(car_protein)

        ahr_atom_1a2 = self.single_net(atom, ahr_protein)
        fused_2b6 = self.multi_net1(atom, pxr_protein, car_protein)
        fused_2c = self.multi_net2(atom, pxr_protein, car_protein)
        fused_3a4 = self.multi_net3(atom, pxr_protein, car_protein)

        ahr_atom_1a2 = ahr_atom_1a2.unsqueeze(1)
        fused_2b6 = fused_2b6.unsqueeze(1)
        fused_2c = fused_2c.unsqueeze(1)
        fused_3a4 = fused_3a4.unsqueeze(1)

        molecule_output = torch.cat([ahr_atom_1a2, fused_2b6, fused_2c, fused_3a4], 1)

        out_share = molecule_output.clone()
        output_layer = self.output_projection(molecule_output).squeeze(-1)
        return out_share, output_layer, None


class ProteinAttentionPooling(nn.Module):
    def __init__(self, dim=1152):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        attn_weights = self.attention(x)
        pooled = torch.sum(attn_weights * x, dim=1)
        return pooled

class GatedMeanPooling(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, atom_feats, batch):
        gates = self.gate_net(atom_feats)
        weighted_feats = atom_feats * gates

        sum_feats = scatter_sum(weighted_feats, batch, dim=0)
        sum_gates = scatter_sum(gates, batch, dim=0)

        sum_gates = sum_gates + 1e-6

        mol_rep = sum_feats / sum_gates

        return mol_rep


class MFiLM(nn.Module):
    def __init__(self, mol_dim=256, prot_dim=1152, hidden_dim=256, dropout=0.1):
        super().__init__()

        def make_film():
            return nn.ModuleDict({
                'gamma': nn.Sequential(
                    nn.Linear(prot_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(hidden_dim, mol_dim), nn.Tanh()
                ),
                'beta': nn.Sequential(
                    nn.Linear(prot_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(hidden_dim, mol_dim), nn.Tanh()
                )
            })

        self.film1 = make_film()
        self.film2 = make_film()
        self.film3 = make_film()

        self.gate = nn.Sequential(
            nn.Linear(mol_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

        self.norm = nn.LayerNorm(mol_dim)

    def forward(self, mol, prot2, prot3):
        B = mol.size(0)
        prot2 = prot2.expand(B, -1)
        prot3 = prot3.expand(B, -1)

        def film_apply(film, prot):
            gamma = film['gamma'](prot)
            beta = film['beta'](prot)
            return (1 + gamma) * mol + beta

        task2 = film_apply(self.film2, prot2)
        task3 = film_apply(self.film3, prot3)

        gate_in = torch.cat([task2, task3], dim=-1)
        w = F.softmax(self.gate(gate_in), dim=-1)
        fused = self.norm(w[:, 0:1] * task2 + w[:, 1:2] * task3)

        return fused

class SFilM(nn.Module):
    def __init__(self, mol_dim=256, prot_dim=1152, hidden_dim=256, dropout=0.1):
        super().__init__()

        def make_film():
            return nn.ModuleDict({
                'gamma': nn.Sequential(
                    nn.Linear(prot_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(hidden_dim, mol_dim), nn.Tanh()
                ),
                'beta': nn.Sequential(
                    nn.Linear(prot_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(hidden_dim, mol_dim), nn.Tanh()
                )
            })

        self.film1 = make_film()
        self.film2 = make_film()
        self.film3 = make_film()

        self.gate = nn.Sequential(
            nn.Linear(mol_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

        self.norm = nn.LayerNorm(mol_dim)

    def forward(self, mol, prot1):
        B = mol.size(0)
        prot1 = prot1.expand(B, -1)

        def film_apply(film, prot):
            gamma = film['gamma'](prot)
            beta = film['beta'](prot)
            return self.norm((1 + gamma) * mol + beta)

        task1 = film_apply(self.film1, prot1)
        return task1


if __name__ == '__main__':
    pass