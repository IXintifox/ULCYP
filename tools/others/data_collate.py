import copy
from tools.cross_data_collate import processed_unlabeled_submodel
from decorator import append
from scipy.special import lmbda
from sklearn.model_selection import train_test_split, StratifiedKFold
from tools.logger import logger
from tools.public_tools import load_config, scaffold_split_no_label
import lmdb
from tools.load_database import load_features_lmdb, load_config
from tqdm import tqdm
import numpy as np
from collections import Counter
from collections import defaultdict
from rdkit.Chem.Scaffolds import MurckoScaffold
import pickle
import joblib

import random
from tools.summerize import summarize_all_datasets, check_self
from tools.UnimolTest import calcaulte_unimol_features
from rdkit import Chem
from tools.data_collate_apr.train_test import (summarize_fold_distribution,
                                               train_test_inducer,
                                               split_labeled_scaffold,
                                               silence_item,
                                               train_test_split,
                                               train2test,
                                               split_unlabeled_test,
                                               split_unlabeled_train_valid)
random.seed(42)

# exclude_test_processed = None

def extract_data(mdb):
    all_data_list = {}
    with mdb.begin(write=False) as txn:
        get_cursor = [itm.decode() for itm in txn.cursor().iternext(values=False)]
        for idx in tqdm(range(len(get_cursor)), total=len(get_cursor)):
            values = pickle.loads(txn.get(str(idx).encode()))
            all_data_list[idx] = values
    return all_data_list


def label_mapping(extract_data_, label_metrix, test_label_mask, fold_train_mask, fold_valid_mask):
    mapping_data = {}


    k = 5
    for i in tqdm(range(len(extract_data_)), total=len(extract_data_), desc="Label mapping"):
        itm = extract_data_[i]
        origin_label = itm["label"]
        origin_label[0] = -1
        itm["label"] = origin_label

        assert (origin_label == label_metrix[i]).all(), "Not consistent"
        itm["test_label_mask"] = test_label_mask[i]
        # itm["train_label_mask"] = train_label_mask[i]
        for f in range(k):
            for subm in range(k):
                itm[f"train_fold_{f}_submodel_{subm}_label_mask"] = fold_train_mask[f][subm][i]
                itm[f"valid_fold_{f}_label_mask"] = fold_valid_mask[f][i]
        mapping_data[i] = itm
    return mapping_data

def cal_unimol(mapping_data):
    keys = list(mapping_data.keys())
    record_smiles = [mapping_data[k]["smiles"] for k in keys]
    unimol_features = calcaulte_unimol_features(record_smiles)


    if not isinstance(unimol_features, np.ndarray):
        unimol_features = np.array(unimol_features)

    for idx, key in enumerate(keys):
        mapping_data[key]["unimol"] = unimol_features[idx]
    return mapping_data



def all_feature_data():  # read
    # mdb = load_features_lmdb()
    # fold_train_submodel_mask, fold_valid_mask, test_label_mask, label_matrix = get_train_test_valid_idx_with_fold(mdb)
    # # check_self(fold_train_submodel_mask)
    # # exit()
    # extract_data_ = extract_data(mdb)
    # joblib.dump(extract_data_, "../DataCenter/extract_data_.pkl")
    # extract_data_ = joblib.load("../DataCenter/extract_data_.pkl")
    #
    #
    # mapping_data = label_mapping(extract_data_, label_matrix, test_label_mask, fold_train_submodel_mask, fold_valid_mask)  # @
    # joblib.dump(mapping_data, "../DataCenter/mapping_data.pkl")
    # mapping_data = joblib.load("../DataCenter/mapping_data.pkl")
    # mapping_data = cal_unimol(mapping_data)
    # joblib.dump(mapping_data, "../DataCenter/mapping_data.pkl")
    # mapping_data = joblib.load("../DataCenter/mapping_data.pkl")
    # label_check_again(mapping_data)
    # joblib.dump(mapping_data, "../DataCenter/mapping_data.pkl")
    # joblib.dump(mapping_data, "../DataCenter/mapping_data.pkl")
    mapping_data = joblib.load("f_30.pkl")
    # exit()
    return mapping_data



def assert_unique_molecules(smiles_list, verbose=True):
    """
    Check for duplicate molecules in SMILES list (based on InChIKey comparison).

    Parameters:
    -------
    smiles_list : list[str]
        Input list of SMILES strings.
    verbose : bool, default=True
        Whether to print detailed information.

    Returns:
    -------
    unique_smiles : list[str]
        Deduplicated SMILES list, order preserved from input.
    """
    inchi_keys = []
    valid_smiles = []
    seen = set()
    duplicates = []

    for i, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError(f"Unable to parse SMILES at index {i}: {s}")
        key = Chem.MolToInchiKey(mol)
        inchi_keys.append(key)
        if key not in seen:
            seen.add(key)
            valid_smiles.append(s)
        else:
            duplicates.append((i, s))

    if len(duplicates) > 0:
        msg = f"Found {len(duplicates)} duplicate molecules (by InChIKey)!"
        if verbose:
            print(msg)
            for i, s in duplicates:
                print(f"  [Duplicate] idx={i} SMILES={s}")
        raise AssertionError(msg)
    else:
        if verbose:
            print(f"No duplicates found, all {len(smiles_list)} molecules are unique.")

    return valid_smiles



def combine_train(fold_train_mask, fold_train_submodel_mask):
    k = 5
    sub_m = 5

    for i in range(k):
        for m in range(sub_m):
            fold_train_submodel_mask[i][m][:, 0:20] = fold_train_mask[i][:, 0:20]
    return fold_train_submodel_mask

def combine_valid(fold_valid_mask, fold_valid_submodel_mask):
    k = 5
    sub_m = 5

    for i in range(k):
        for m in range(sub_m):
            fold_valid_submodel_mask[i][:, 0:20] = fold_valid_mask[i][:, 0:20]
    return fold_valid_submodel_mask


def get_train_test_valid_idx_with_fold(mdb):
    # Fast load
    label_dict = {}
    dataset_dict = {}
    smiles_dict = {}
    with mdb.begin(write=False) as txn:
        get_cursor = [itm.decode() for itm in txn.cursor().iternext(values=False)]
        for idx in tqdm(range(len(get_cursor)), total=len(get_cursor)):
            values = pickle.loads(txn.get(str(idx).encode()))
            label_dict[idx] = values["label"]
            dataset_dict[idx] = values["datasets"]
            assert len(values["datasets"]) == np.sum(values["label"]!=-1), f"error{idx}"


            smiles_dict[idx] = values["smiles"]
    joblib.dump(label_dict, "label_dict.pkl")
    joblib.dump(dataset_dict, "dataset_dict.pkl")
    joblib.dump(smiles_dict, "smiles_dict.pkl")

    label_dict = joblib.load("label_dict.pkl")
    dataset_dict = joblib.load("dataset_dict.pkl")
    smiles_dict = joblib.load("smiles_dict.pkl")



    label_matrix, origin_train_label_mask, origin_test_label_mask, smiles_list = train_test_tidy(label_dict, dataset_dict,
                                                                                   smiles_dict)

    task_mask = origin_train_label_mask + origin_test_label_mask # check?

    # print(np.where(label_matrix[:, 1:2][label_matrix[:,  1:2] > -1]))
    # print(np.where(task_mask[:,  1:2][task_mask[:,  1:2] == 1]))

    assert ((label_matrix != -1) == ([task_mask == 1])).all(), "?"
    # print((label_matrix[label_matrix != -1] == task_mask[task_mask == 1]).all())
    train_label_mask_origin = copy.deepcopy(origin_train_label_mask)
    task_mask = origin_train_label_mask + origin_test_label_mask


    train_label_mask, test_label_mask, label_matrix = silence_item(origin_train_label_mask, origin_test_label_mask, label_matrix, smiles_list)
    tt = train_label_mask + test_label_mask

    # Split inducer samples
    train_label_mask, test_label_mask = train_test_inducer(train_label_mask,test_label_mask, smiles_list)
    # Split PXR CAR samples
    train_label_mask, test_label_mask = train_test_split(train_label_mask, test_label_mask, label_matrix, smiles_list)
    # Move overlapping samples with inducers to test set
    train_label_mask, test_label_mask = train2test(train_label_mask, test_label_mask)
    # At this point, train/test sets are strictly split, missing unlabeled data
    assert (tt == (train_label_mask+test_label_mask)).all(), "Consistant"

    # Split unlabeled data for test set
    test_label_mask = split_unlabeled_test(train_label_mask, test_label_mask, smiles_list)
    fold_valid, fold_train = split_unlabeled_train_valid(train_label_mask, test_label_mask, smiles_list)
    # 5-fold cross validation for inducers
    summarize_fold_distribution(fold_valid, fold_train, label_matrix)  # for testing
    # 5-fold cross validation for other samples
    other_fold_train, other_fold_valid = split_labeled_scaffold(train_label_mask, label_matrix, smiles_list)
    # Merge
    fold_train_submodel_mask = combine_train(other_fold_train, fold_train)
    fold_valid_mask = combine_valid(other_fold_valid, fold_valid)


    return fold_train_submodel_mask, fold_valid_mask, test_label_mask, label_matrix



def train_test_tidy(label_dict, dataset_dict, smiles_dict): 
    dataset2bitdict = {i: idx for idx, i in enumerate(load_config()["vector_bit"])}

    train_label_mask = []
    test_label_mask = []
    label_matrix = []
    smiles_list = []
    for idx in tqdm(range(len(label_dict)), total=len(label_dict)):

        assert len(dataset_dict[idx]) == np.sum(label_dict[idx] != -1), f"error{idx}"

        training_label_m = np.zeros(len(label_dict[0]))
        test_label_m = np.zeros(len(label_dict[0]))

        label_matrix.append(label_dict[idx])
        smiles_list.append(smiles_dict[idx])
        datasets = dataset_dict[idx]
        for d in datasets:
            d_subtask = d.split("_")[0] + "_" + "".join([d.split("_")[1][0].lower(), d.split("_")[1][1:]]) + "_" + \
                        d.split("_")[2]
            if "Train" in d_subtask:
                task_name = d_subtask.replace("_Train", "")
                training_label_m[dataset2bitdict[task_name]] = 1
            elif "Test" in d_subtask:
                task_name = d_subtask.replace("_Test", "")
                test_label_m[dataset2bitdict[task_name]] = 1
            elif "Valid" in d_subtask:
                task_name = d_subtask.replace("_Valid", "")
                training_label_m[dataset2bitdict[task_name]] = 1
            else:
                raise NotImplementedError

        train_label_mask.append(training_label_m)
        test_label_mask.append(test_label_m)

        full_tmp_mask = training_label_m + test_label_m
        assert ((full_tmp_mask==1) == (label_dict[idx] != -1)).all(), f"error_3333_{idx}"




    label_matrix = np.array(label_matrix)
    train_label_mask = np.array(train_label_mask)
    test_label_mask = np.array(test_label_mask)

    # check
    full_mask = train_label_mask + test_label_mask
    test_matrix = copy.deepcopy(label_matrix)
    test_matrix[full_mask == 1] = -2
    for idx in tqdm(range(len(label_dict)), total=len(label_dict)):
        assert np.max(test_matrix[idx]) == -1, idx

    assert np.max(full_mask) <= 1, "Check-pass1"  # No overlapping masks
    assert np.max(test_matrix) == -1, "Check-pass2"  # Full coverage, no 0 or 1
    return label_matrix, train_label_mask, test_label_mask, smiles_list


if __name__ == '__main__':
    pass