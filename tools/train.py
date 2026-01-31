import os
import pickle

import numpy as np
import tqdm
from torch_geometric.loader import DataLoader
from tools.graph_dataloader import DataCollate
from tools.output_results import log_epoch_metrics_to_csv
from tools.data_representation import dataset_map
from tools.logger import logger
from tools import data_collate
from tools.graph_dataloader import DataLoadingTools
from tools.load_protein import load_protein
from tools.metrics import Record
from ulcyp.backbone import ULCYP
from tools.calculate_prior import calculate_prior
import torch
import joblib
import random
# from ulcyp.utils import protein_process
import copy
import os, random, numpy as np, torch
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
import umap.umap_ as umap

if os.path.exists("features") is False:
    os.mkdir("features")

    
def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(712)


def start(fold, dataset_cover):
    if not os.path.exists(f"model_save/{fold}"):
        os.makedirs(f"model_save/{fold}")

    model_checkpoint = None

    test_dataset = DataCollate(dataset_cover, name="test")

    follow_batch = ["unimol_atom"]
    exclude_key = []

    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True,
                                 #num_workers=16,
                                 follow_batch=follow_batch, exclude_keys=exclude_key)


    valid_dataset_pu = DataCollate(dataset_cover, name="valid", fold=fold, only_pu=True)
    valid_pu_dataloader = DataLoader(valid_dataset_pu, batch_size=128, shuffle=True,
                                     #num_workers=16,
                                     follow_batch=follow_batch, exclude_keys=exclude_key)

    print(len(valid_dataset_pu))
    cluster_number = 5

    train_pu_dataloader_dict = {}
    for sub_model in range(cluster_number):
        train_dataset_pu = DataCollate(dataset_cover, name="train", fold=fold, sub_model_num=sub_model)
        train_pu_dataloader = DataLoader(train_dataset_pu, batch_size=128, shuffle=True,
                                         # num_workers=16,
                                      follow_batch=follow_batch, exclude_keys=exclude_key)
        train_pu_dataloader_dict[sub_model] = train_pu_dataloader


    device = "cuda"
    protein = load_protein()
    if model_checkpoint is not None:
        print(f"Loading model checkpoint: {model_checkpoint}")
        model = ULCYP(device=device, sub_model_num=cluster_number).to(device)
        model.load_state_dict(torch.load(model_checkpoint))
    else:
        model = ULCYP(device=device, sub_model_num=cluster_number).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    ema_model = copy.deepcopy(model)
    ema_decay = 0.999
    for param in ema_model.parameters():
        param.requires_grad = False

    small_params, other_params = [], []
    for name, param in model.named_parameters():
        if "down_stream_task.inducer_model_seq" in name:
            small_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.Adam([
        {"params": other_params, "lr": 1e-4},
        {"params": small_params, "lr": 1e-4},
    ], betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)

    optimizer_update = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-5)

    start_save_file_epoch = 200
    pu_start_epoch = 0
    record_tools = Record(sub_model_num=cluster_number)
    best_va_pr_auc = 0
    epoch = 150
    for epoch in range(epoch):

        train_data = {"train_dataloader": train_dataloader, "train_pu_dataloader":train_pu_dataloader_dict, "protein": protein}
        valid_data = {"valid_dataloader": valid_dataloader,"train_pu_dataloader":valid_pu_dataloader,  "protein": protein}
        test_data = {"test_dataloader": test_dataloader, "protein": protein}
        train(epoch, train_data, model, ema_model, record_tools, optimizer, fold=fold, pu_start_epoch=pu_start_epoch)
        return_valid_auc = valid(epoch, valid_data, model, record_tools, fold=fold, pu_start_epoch=pu_start_epoch, used_model=cluster_number)
        # return_test_auc = test(epoch, test_data, model, record_tools,  pu_start_epoch=pu_start_epoch, used_model=cluster_number)


        _, valid_prauc_pu_epoch = get_mean_auc(return_valid_auc)
        # _, test_prauc_pu_epoch = get_mean_auc(return_test_auc)

        if valid_prauc_pu_epoch > best_va_pr_auc:
            best_va_pr_auc = valid_prauc_pu_epoch
            if epoch > start_save_file_epoch:
                file_name = f"model_save/{fold}/{epoch}_valid_pu_aur_best.pt"
                torch.save([], file_name)

        optimizer_update.step()

        record = record_tools.to_dict()
        joblib.dump(record, f"model_save/{fold}/record_summary.joblib")

def train(epoch, train_data, model, ema_model, record_tools, optimizer, fold=None, pu_start_epoch=0):
    print("Train epoch: ", epoch)
    model.train()

    train_dataloader, train_pu_dataloader_dict, protein = train_data.values()
    sub_task_number = len(train_pu_dataloader_dict.keys())


    for task_number in tqdm.tqdm(list(train_pu_dataloader_dict.keys()), total=sub_task_number, desc="train_pu_model"):
        train_pu_dataloader = train_pu_dataloader_dict[task_number]

        for idx, i in tqdm.tqdm(enumerate(train_pu_dataloader), total=len(train_pu_dataloader), disable=True, desc="train_pu"):
            input_data = {"molecule": i, "protein": protein}
            model_return = model.get_loss(input_data, dataset_name="train", pu_task=True, f=fold, cluster=task_number)
            loss = model_return["loss"]

            if loss != 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                record_tools.update_train_loss(epoch, loss.item())
                if idx % 50 == 0:
                    print("training loss", loss.item())
            label_idx = list(range(model_return["label"].shape[-1]))
            pred, label, mask = (model_return["pred"].detach().cpu().numpy()
                                     , model_return["label"][:, label_idx].detach().cpu().numpy(),
                                 model_return["mask"][:, label_idx].detach().cpu().numpy())


            record_tools.update_iter_results(epoch, label, pred, mask, phase="train_pu")

        print(f"epoch:{epoch} - {np.mean(gate1)} - {1-np.mean(gate1)}")


    if epoch == pu_start_epoch:
        prior_dict = {}
        for task_number in tqdm.tqdm(list(train_pu_dataloader_dict.keys()), total=len(train_pu_dataloader_dict),
                                     desc="train_pu_model"):

            record_prior_embedding = []
            record_prior_mask = []
            record_prior_label = []

            train_pu_dataloader = train_pu_dataloader_dict[task_number]

            for idx, i in tqdm.tqdm(enumerate(train_pu_dataloader), total=len(train_pu_dataloader), disable=False,
                                    desc="train_for_prior_calculate"):
                input_data = {"molecule": i, "protein": protein}
                # model.get_loss(input_data, dataset_name="train", pu_task=False, f=fold, cluster=task_number)
                label = input_data["molecule"]["label"]
                mask = input_data["molecule"][f"train_fold_{fold}_submodel_{task_number}_label_mask"]
                finger_features = i["fingerprint6"].unsqueeze(1)
                # record_prior_embedding.append(model.downstream_embedding.detach().cpu().numpy())
                record_prior_embedding.append(finger_features.expand(-1, 20+6*sub_task_number, -1))
                record_prior_mask.append(mask.detach().numpy())
                record_prior_label.append(label.detach().numpy())


            record_prior_cal = np.concatenate(record_prior_embedding, axis=0)
            record_prior_mask = np.concatenate(record_prior_mask, axis=0)
            record_prior_label = np.concatenate(record_prior_label, axis=0)

            prior = calculate_prior(record_prior_cal, record_prior_mask, record_prior_label, sub_task_number, task_number)
            prior_dict[task_number] = prior

        model.loss_tools.prior = prior_dict
        ema_model.loss_tools.prior = prior_dict
        record_tools.prior = prior_dict

    for p, ema_p in zip(model.parameters(), ema_model.parameters()):
        ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)



def valid(epoch, valid_data, model, record_tools, fold, pu_start_epoch=0, used_model=1):
    model.eval()
    valid_dataloader, valid_pu_dataloader, protein = valid_data.values()
    record_all_loss = []

    if epoch > pu_start_epoch:
        print("#### Valid PU model")
        for idx, i in tqdm.tqdm(enumerate(valid_pu_dataloader), total=len(valid_pu_dataloader), disable=False,
                                desc="valid_pu"):
            input_data = {"molecule": i, "protein": protein}
            model_return = model.get_loss(input_data, dataset_name="valid", pu_task=True, f=fold)

            loss = model_return["loss"]
            record_tools.update_valid_pu_loss(epoch, loss.item())
            label_idx = list(range(model_return["label"].shape[-1]))
            pred, label, mask = (model_return["pred"].detach().cpu().numpy()
                                     , model_return["label"][:, label_idx].detach().cpu().numpy(),
                                 model_return["mask"][:, label_idx].detach().cpu().numpy())

            record_tools.update_iter_results(epoch, label, pred, mask, phase="valid_pu")
            if idx % 20 == 0:
                print("valid loss", loss.item())

    record_tools.cal_metrics(epoch, phase="valid", is_pu=False)
    if epoch > pu_start_epoch:
        record_tools.cal_metrics(epoch, phase="valid_pu", is_pu=True, used_model=used_model)
        record_tools.cal_pu_vote_metrics(epoch=epoch, phase="valid_pu", used_model=used_model)
        metrics_epoch_valid_pu = record_tools.get_metrics(epoch=epoch, phase="valid_pu")
        metrics_epoch_valid_concat_pu = record_tools.get_metrics(epoch=epoch, phase="valid_concat_pu")
    else:
        metrics_epoch_valid_pu = None
        metrics_epoch_valid_concat_pu = None
    metrics_epoch_valid = record_tools.get_metrics(epoch=epoch, phase="valid")

    return metrics_epoch_valid, metrics_epoch_valid_concat_pu


def test(epoch, test_data, model, record_tools, pu_start_epoch, used_model=1):
    model.eval()
    test_dataloader, protein = test_data.values()

    for idx, i in enumerate(test_dataloader):
        input_data = {"molecule": i, "protein": protein}
        model_return = model.predict(input_data)
        label, mask = i["label"], i["test_label_mask"]
        label_idx = list(range(label.shape[-1]))
        pred, label, mask = (model_return["pred"].detach().cpu().numpy()
                                 , label[:, label_idx].detach().cpu().numpy(),
                             mask[:, label_idx].detach().cpu().numpy())

        record_tools.update_iter_results(epoch, label, pred, mask, phase="test_pu")

    record_tools.cal_metrics(epoch, phase="test", is_pu=False)
    if epoch > pu_start_epoch:
        record_tools.cal_metrics(epoch, phase="test_pu", is_pu=True, used_model=used_model)
        record_tools.cal_pu_vote_metrics(epoch=epoch, phase="test_pu", used_model=used_model)
        metrics_epoch_test_pu = record_tools.get_metrics(epoch=epoch, phase="test_pu")
        metrics_epoch_test_concat_pu = record_tools.get_metrics(epoch=epoch, phase="test_concat_pu")
    else:
        metrics_epoch_test_pu = None
        metrics_epoch_test_concat_pu = None
    metrics_epoch_test = record_tools.get_metrics(epoch=epoch, phase="test")

    return metrics_epoch_test, metrics_epoch_test_concat_pu

def get_mean_auc(results):
    normal_epoch, normal_pu_epoch = results

    def get_metric(extract_metric, metric="AUR_norm"):
        record_metric = []
        for key in list(extract_metric.keys()):
            record_metric.append(extract_metric[key][metric])
        return np.mean(record_metric)

    if normal_epoch is not None:
        normal_return = get_metric(normal_epoch)
    else:
        normal_return = 0

    if normal_pu_epoch is not None:
        pu_return =  get_metric(normal_pu_epoch)
    else:
        pu_return = 0

    return normal_return, pu_return


if __name__ == '__main__':
    # dataset_map()
    #
    label_mapping_data = data_collate.all_feature_data()

    graph_data_tools = DataLoadingTools()
    graph_data_tools.get_data(label_mapping_data, list(range(len(label_mapping_data))))
    dataset_cover = graph_data_tools.data_cover


    for fold in range(0, 1):
        start(fold, dataset_cover)