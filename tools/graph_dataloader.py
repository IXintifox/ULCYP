from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np
import torch

class BasedData(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class DataLoadingTools:
    def __init__(self):
        self.data_cover = []

    def get_data(self, full_data_dict, key_id):

        for idx in tqdm(key_id, total=len(key_id), desc="DataLoading to DataBase"):
            update_values = full_data_dict[idx]
            update_values.pop("datasets", 0)
            update_values.pop("inchi_key", 0)
            update_values.pop("graph_with_h", 0)
            update_values.pop("graph_no_h", 0)

            for k in list(update_values.keys()):
                if "label" in k:
                    update_values[k] = torch.from_numpy(update_values[k][:, ]).reshape(1, -1)[:, 18:22]
                if "mask" in k:
                    update_values[k] = torch.from_numpy(np.array(update_values[k]))[:, 18:22]

                if (k == "fingerprint4") or (k == "fingerprint6"):
                    if len(np.array(update_values[k]).shape) == 1:
                        if not isinstance(update_values[k], np.ndarray):
                            update_values[k] = torch.from_numpy(np.array(update_values[k])).reshape(1, -1)
                        else:
                            update_values[k] = torch.from_numpy(update_values[k])
                if k in ["unimol", "gemini"]:
                    update_values[k] = torch.from_numpy(np.array(update_values[k])).reshape(1, -1)

                elif k not in ["smiles", "datasets", "mol"]:
                    if not isinstance(update_values[k], np.ndarray):
                        update_values[k] = torch.from_numpy(np.array(update_values[k]))
                    else:
                        update_values[k] = torch.from_numpy(update_values[k])

            self.data_cover.append(BasedData(**update_values))


class DataCollate(Dataset):
    def __init__(self, dataset, name=None, fold=None, sub_model_num=None, full_fold=None, all_sub_model_num=None):
        self.name = name
        assert name is not None
        self.fold = fold
        self.sub_model_num = sub_model_num
        self.dataset = dataset
        self.all_full_fold = full_fold
        self.all_sub_model_num = all_sub_model_num
        self.data = []
        self.screen_dataset(name)

    def screen_dataset(self, name):
        if name == "prediction":
            for itm in self.dataset:
                self.data.append(itm)
        if name == "test":
            for itm in self.dataset:
                label_task = itm.test_label_mask
                if torch.sum(label_task) > 0:
                    self.data.append(itm)
        elif name == "train":
            assert self.fold is not None
            for itm in self.dataset:
                label_task = getattr(itm, f"train_fold_{self.fold}_submodel_{self.sub_model_num}_label_mask")
                if len(label_task.shape) == 2:
                    label_task = label_task.squeeze(0)
                if torch.sum(label_task) > 0:
                    self.data.append(itm)

        elif name == "valid":
            assert self.fold is not None
            for itm in self.dataset:
                label_task = getattr(itm, f"valid_fold_{self.fold}_label_mask")
                if len(label_task.shape) == 2:
                    label_task = label_task.squeeze(0)
                if torch.sum(label_task) > 0:
                    self.data.append(itm)


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
