from collections import defaultdict
import torch.nn as nn
import torch
from torch_scatter import (scatter_mean,
                           scatter_max)
from ulcyp.downstream import (DownStreamTask,
                              TaskClass)
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from captum.attr import IntegratedGradients


class ULCYP(nn.Module):
    def __init__(self, hidden_channels=256, out_channels=256,
                 protein_input_dim=1152, prior=None, sub_model_num=2, device="cpu"):
        super(ULCYP, self).__init__()
        self.prior = prior
        self.loss_tools = CalculateLoss(prior=self.prior, sub_model_num=sub_model_num, device=device)
        self.fd_tools = DesFinger(hidden_dim=hidden_channels)
        self.down_stream_task = DownStreamTask(protein_input_dim=protein_input_dim,
                                               protein_hidden_dim=hidden_channels,
                                               hidden_dim=out_channels,
                                               sub_model_num=sub_model_num,
                                               device=device)

        self.downstream_embedding = None
        self.downstream_mask = None
        self.upstream_embedding = None
        self.device = device
        self.sigmoid = nn.Sigmoid()
        self.gate_fusion = GatedFusion()
        self.fusion = FusionMLP()

        self.gate_layer = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.Sigmoid()
        )

    @staticmethod
    def logit_combine(features):
        features = features[:, 18:]
        reshaped = features.view(features.size(0), 5, 4)
        return torch.mean(reshaped, dim=1)

    def ig_forward(self, unimol_features, gemini_features, des, protein):
        des, unimol_features, gemini_features = self.fd_tools(des, unimol_features, gemini_features)
        self.upstream_embedding_unimol = torch.mean(unimol_features,  dim=0, keepdim=True)
        self.upstream_embedding_gemini = gemini_features

        self.upstream_embedding, gt = self.gate_fusion(self.upstream_embedding_gemini, self.upstream_embedding_unimol)
        self.gate1 = torch.mean(gt).item()

        res_fused = self.upstream_embedding + des
        gate_input = torch.cat([res_fused, self.upstream_embedding], dim=-1)
        gate = re_calibrate_output(self.gate_layer(gate_input))
        fused_final = gate * res_fused + (1 - gate) * self.upstream_embedding
        self.gate2 = torch.mean(gate).item()
        fused_final = self.fusion(fused_final)
        (output_features_cover, _, _) = self.down_stream_task(fused_final, protein, None)
        output_features_cover = self.logit_combine(output_features_cover)
        return output_features_cover


    def ig_wrapper(self, unimol_features, gemini_features, des, protein):
        steps = unimol_features.shape[0]
        outputs = []
        for i in range(steps):
            step_feat = unimol_features[i]
            out = self.ig_forward(step_feat, gemini_features[i].unsqueeze(0), des[i].unsqueeze(0), protein)
            outputs.append(out)
        return torch.cat(outputs, dim=0)

    def ig_predict(self, x):
        molecule, protein = x["molecule"], x["protein"]
        unimol_features = molecule.unimol_atom.to(self.device)
        unimol_batch = molecule.unimol_atom_batch.to(self.device)
        gemini_features = molecule.gemini.to(self.device)
        descriptor = molecule.descriptors.to(self.device).float()
        descriptor = des_norm(descriptor)

        batch_num = gemini_features.shape[0]
        batch_results = []

        ig = IntegratedGradients(self.ig_wrapper)


        for b in range(batch_num):
            b_unimol_features = unimol_features[unimol_batch == b].unsqueeze(0)
            b_gemini_features = gemini_features[b].unsqueeze(0)
            b_descriptor = descriptor[b].unsqueeze(0)


            input_tensor = b_unimol_features.detach().clone()
            input_tensor.requires_grad = True

            all_task = []
            for task_id in range(4):
                attributions = ig.attribute(
                    inputs=input_tensor,
                    target=task_id,
                    n_steps=50,
                    additional_forward_args=(b_gemini_features, b_descriptor, protein)
                )
                attributions = torch.sum(attributions, dim=-1, keepdim=True).detach().cpu()
                all_task += attributions
            all_task = torch.cat(all_task, dim=-1).numpy()
            batch_results.append(all_task)

        return batch_results


    def forward(self, x):
        molecule, protein = x["molecule"], x["protein"]

        unimol_features = molecule.unimol_atom.to(self.device)
        unimol_batch = molecule.unimol_atom_batch.to(self.device)

        gemini_features = molecule.gemini.to(self.device)

        descriptor = molecule.descriptors.to(self.device).float()
        descriptor = des_norm(descriptor)
        des, unimol_features, gemini_features = self.fd_tools(descriptor, unimol_features, gemini_features)

        self.upstream_embedding_unimol = scatter_mean(unimol_features, unimol_batch, dim=0)
        self.upstream_embedding_gemini = gemini_features

        self.upstream_embedding, gt = self.gate_fusion(self.upstream_embedding_gemini, self.upstream_embedding_unimol)
        self.gate1 = torch.mean(gt).item()


        res_fused = self.upstream_embedding + des
        gate_input = torch.cat([res_fused, self.upstream_embedding], dim=-1)  # [B, 1536]
        gate = re_calibrate_output(self.gate_layer(gate_input))  # [B, 768]
        fused_final = gate * res_fused + (1 - gate) * self.upstream_embedding
        self.gate2 = torch.mean(gate).item()
        fused_final = self.fusion(fused_final)
        (output_features_cover, molecule_features_cover,
         _) = self.down_stream_task(fused_final, protein, unimol_batch)
        return output_features_cover, molecule_features_cover, ""

    def get_loss(self, x, dataset_name, f=0, pu_task=False, training=False, cluster=1):
        label = x["molecule"]["label"]
        test_mask = x["molecule"]["test_label_mask"]
        train_mask = x["molecule"][f"train_fold_{f}_submodel_{cluster}_label_mask"]
        valid_mask = x["molecule"][f"valid_fold_{f}_label_mask"]

        if dataset_name == "train":
            mask = train_mask
        elif dataset_name == "valid":
            mask = valid_mask
        elif dataset_name == "test":
            mask = test_mask
        else:
            raise ValueError("dataset_name must be 'train' or 'valid' or 'test'")

        output_features_cover, molecule_features_cover, _ = self(x)

        label = label.to(self.device)
        mask = mask.to(self.device)

        self.downstream_embedding = molecule_features_cover
        self.downstream_mask = mask
        self.downstream_label = label

        con_loss = contrastive_loss_from_features(self.upstream_embedding_unimol, self.upstream_embedding_gemini)
        pu_task_loss = self.loss_tools.get_pu_task_loss(output_features_cover, label, mask, cluster=cluster)

        full_loss = pu_task_loss + 0.1 * con_loss
        return_results = {
            "pu_loss": pu_task_loss,
            "loss": full_loss,
            "pred": self.sigmoid(output_features_cover),
            "label": label,
            "mask": mask,
        }
        return return_results

    def get_temperature(self, device, high_temp=2.0, low_temp=1.0):
        cycle_unit = torch.tensor([low_temp, 1.5, 3, high_temp])
        part_tasks = cycle_unit.repeat(5)
        full_temperature = part_tasks
        return full_temperature.to(device)


    def predict(self, x):
        temp = self.get_temperature(self.device)
        output_features_cover, molecule_features_cover, _ = self(x)
        print(output_features_cover.shape, molecule_features_cover.shape)
        return {
            "pred": self.sigmoid(output_features_cover/temp),
            "embedding": molecule_features_cover,
        }

    def set_requires_grad(self, module, requires_grad: bool):
        for p in module.parameters():
            p.requires_grad = requires_grad

class GatedFusion(nn.Module):
    def __init__(self, dim_a=256, dim_b=256, dim_hidden=256):
        super(GatedFusion, self).__init__()
        self.proj_a = nn.Linear(dim_a, dim_hidden)
        self.proj_b = nn.Linear(dim_b, dim_hidden)
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim_hidden * 2, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, 1),
            nn.Sigmoid()
        )
        nn.init.constant_(self.gate_mlp[-2].bias, 0.0)

    def forward(self, feat_a, feat_b):
        a_proj = self.proj_a(feat_a)
        b_proj = self.proj_b(feat_b)
        concat = torch.cat([a_proj, b_proj], dim=-1)
        gate = self.gate_mlp(concat)
        gate = F.dropout(gate, p=0.2)
        fused = gate * a_proj + (1 - gate) * b_proj
        return fused, gate


class DesFinger(nn.Module):
    def __init__(self,hidden_dim, dropout=0.1, device="cpu"):
        super(DesFinger, self).__init__()
        self.fingerprint_dim = 2048
        self.des_dim = 217
        self.uni_dim = 768
        self.gemini_dim = 2048
        self.ourb_dim = 2048
        self.middle_dim = 1024+512
        self.middle_dim2 = 1024
        self.hidden_dim = 256

        self.des_mlp = nn.Sequential(
            nn.LayerNorm(self.des_dim),
            nn.Linear(self.des_dim, self.hidden_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim//2,self.hidden_dim*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim*2,self.hidden_dim),
            nn.Tanh()
        )

        self.uni_mlp = nn.Sequential(
            nn.Linear(self.uni_dim, self.uni_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.uni_dim//2,self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim,self.hidden_dim),
        )

        self.gemini_mlp = nn.Sequential(
            nn.Linear(self.gemini_dim, self.middle_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.middle_dim,self.middle_dim2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.middle_dim2,self.middle_dim2//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.middle_dim2//2, self.hidden_dim),
        )

    def forward(self, descriptor, unimol_features, gemini_features):
        descriptor = torch.nan_to_num(descriptor, nan=0.0)
        return  self.des_mlp(descriptor), self.uni_mlp(unimol_features), self.gemini_mlp(gemini_features)

class CalculateLoss(TaskClass):
    def __init__(self, prior, sub_model_num=1, device="cuda"):
        super().__init__(sub_model_num)
        self.normal_loss_function = BCEWithLogitsLoss(reduction="none")
        self.device = device
        self.pu_learning = PULoss(device=device)
        self.prior = prior
        self.positive_pool = defaultdict(dict)

    def get_pu_task_loss(self, pred, label, mask, cluster):
        pred_idx = self.loop_special_task_idx[cluster]
        prior = self.prior[cluster]

        label = label
        mask = mask.float()
        pred = pred[:, pred_idx]   #

        total_loss = 0.0
        valid_batch = 0

        for task in range(len(self.pu_task_idx)):

            single_label = label[:,task][mask[:,task] == 1]
            single_pred = pred[:,task][mask[:,task] == 1]

            positive = single_pred[single_label == 1]
            unlabel = single_pred[single_label == -1]


            if len(positive) > 0:
                self.positive_pool[cluster][task] = positive.detach().clone()
            elif task in self.positive_pool[cluster]:
                positive = self.positive_pool[cluster][task]
            else:
                continue

            if len(positive) == 0 or len(unlabel) == 0:
                continue
            task_loss = self.pu_learning(prior[task], positive, unlabel)
            if task_loss != 0:
                valid_batch += 1

            if not task_loss.requires_grad:
                task_loss = task_loss + 0.0 * pred.mean()
            total_loss += task_loss
        return total_loss

    def supervised_contrastive_loss(self, features, labels, temperature=0.07):
        features = F.normalize(features, p=2, dim=1)
        labels = labels.contiguous().view(-1, 1)
        batch_size = features.shape[0]

        mask = torch.eq(labels, labels.T).float().to(self.device)

        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=self.device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)

        loss = -mean_log_prob_pos.mean()
        return loss


class PULoss(nn.Module):
    def __init__(self, gamma: float = 1.0, beta: float = 0.0, nnPU: bool = True, eps: float = 1e-6, device: str = "cuda"):
        super().__init__()
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.nnPU = bool(nnPU)
        self.eps = float(eps)
        self.device = device

    def forward(self, prior: float, outputs_p: torch.Tensor, outputs_u: torch.Tensor):


        ones_p = torch.ones_like(outputs_p, device=outputs_p.device)
        zeros_p = torch.zeros_like(outputs_p, device=outputs_p.device)
        zeros_u = torch.zeros_like(outputs_u, device=outputs_u.device)

        loss_pos = F.binary_cross_entropy_with_logits(outputs_p, ones_p, reduction='mean')
        loss_neg_on_p = F.binary_cross_entropy_with_logits(outputs_p, zeros_p, reduction='mean')
        loss_unlabeled = F.binary_cross_entropy_with_logits(outputs_u, zeros_u, reduction='mean')

        pi = float(max(1e-6, min(1.0 - 1e-6, prior)))
        positive_risk = pi * loss_pos
        negative_risk = loss_unlabeled - pi * loss_neg_on_p

        if self.nnPU and negative_risk.item() < -self.beta:
            total_loss = self.gamma * (positive_risk - self.beta)
        else:
            total_loss = positive_risk + self.gamma * negative_risk

        return total_loss

def des_norm(descriptor):
    nan_mask = torch.isnan(descriptor)
    inf_mask = torch.isinf(descriptor)
    col_mean = torch.nanmean(descriptor, dim=0).unsqueeze(0).expand(nan_mask.shape)
    descriptor[nan_mask] = col_mean[nan_mask]
    descriptor[inf_mask] = col_mean[inf_mask]
    epsilon = 1e-8
    log_descriptor = torch.log(torch.abs(descriptor) + epsilon)
    log_descriptor = torch.where(descriptor < 0, -log_descriptor, log_descriptor)
    return log_descriptor

def re_calibrate_output(x):
    return torch.sigmoid(x)

def contrastive_loss_from_features(f1, f2, tau=0.1):
    f1 = F.normalize(f1, p=2, dim=1)
    f2 = F.normalize(f2, p=2, dim=1)
    logits = (f1 @ f2.t()) / tau
    N = f1.size(0)
    labels = torch.arange(N, dtype=torch.long, device=f1.device)
    loss_f1_f2 = F.cross_entropy(logits, labels)
    loss_f2_f1 = F.cross_entropy(logits.t(), labels)
    loss = (loss_f1_f2 + loss_f2_f1) / 2
    return loss


class FusionMLP(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, output_dim=256, dropout=0.3):
        super(FusionMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        return x
