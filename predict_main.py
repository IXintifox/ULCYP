import numpy as np
import torch
from torch_scatter import scatter_mean, scatter_max
from collections import defaultdict
from tools.data_representation_for_test import data_representation
from tools.graph_dataloader import DataLoadingTools
from tools.graph_dataloader import DataCollate
from torch_geometric.loader import DataLoader
from tools.load_protein import load_protein
from ulcyp.backbone import ULCYP
from sklearn.metrics import auc
import matplotlib.cm as cm
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import SimilarityMaps
import matplotlib.pyplot as plt
import os
import io
from PIL import Image
from IPython.display import SVG, display
import random
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem import AllChem
from unimol_tools.data.conformer import create_mol_from_atoms_and_coords
import base64
import pandas as pd
from rdkit.Chem import DataStructs
from rdkit.Chem.AllChem import GetMorganGenerator
from tools.domain import calculate_domain_levels
from tools.predict_config_loader import (
    load_predict_config,
    get_device_from_config,
    get_model_params_from_config,
    get_prediction_params_from_config
)

fp_gen = GetMorganGenerator(radius=2, includeChirality=True)


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


setup_seed(7)


def main_with_var(smiles, config_path="config.yaml", cal_ig=None, T=50):
    config = load_predict_config(config_path)
    device = get_device_from_config(config)
    model_params = get_model_params_from_config(config)
    pred_params = get_prediction_params_from_config(config)

    processed_data = data_representation(smiles)
    key_id = processed_data.keys()
    graph_data_tools = DataLoadingTools()
    graph_data_tools.get_data(processed_data, key_id)
    dataset_cover = graph_data_tools.data_cover

    protein = load_protein()

    test_dataset = DataCollate(dataset_cover, name="prediction")
    follow_batch = ["unimol_atom"]

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=pred_params['batch_size'],
        shuffle=False,
        follow_batch=follow_batch,
        exclude_keys=[]
    )

    print("load_model", pred_params['model_checkpoint'])
    model = ULCYP(**model_params).to(device)
    model.load_state_dict(torch.load(pred_params['model_checkpoint']))

    test_data = {"test_dataloader": test_dataloader, "protein": protein}
    record_range = []

    for t in range(T):
        if cal_ig and (t == (T-1)):
            pred, ig = test_for_var(test_data, model, cal_ig=True)
        else:
            pred, ig = test_for_var(test_data, model, cal_ig=False)
        pred = results_extract(pred)
        record_range.append(pred)

    get_mol = molecule_recon(test_dataset, ig) if cal_ig else None
    return record_range, list(key_id), get_mol


def main(smiles, config_path="config.yaml", cal_ig=None):
    config = load_predict_config(config_path)
    device = get_device_from_config(config)
    model_params = get_model_params_from_config(config)
    pred_params = get_prediction_params_from_config(config)

    processed_data = data_representation(smiles)
    key_id = processed_data.keys()
    graph_data_tools = DataLoadingTools()
    graph_data_tools.get_data(processed_data, key_id)
    dataset_cover = graph_data_tools.data_cover

    protein = load_protein()

    test_dataset = DataCollate(dataset_cover, name="prediction")
    follow_batch = ["unimol_atom"]

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=pred_params['batch_size'],
        shuffle=False,
        follow_batch=follow_batch,
        exclude_keys=[]
    )

    model = ULCYP(**model_params).to(device)
    model.load_state_dict(torch.load(pred_params['model_checkpoint']))

    test_data = {"test_dataloader": test_dataloader, "protein": protein}
    pred, ig = test(test_data, model, cal_ig=cal_ig)
    pred = results_extract(pred)
    get_mol = molecule_recon(test_dataset, ig) if cal_ig else None
    return pred, list(key_id), get_mol


def molecule_recon(data_cover, ig):
    data_num = len(data_cover)
    record_mol = []
    for idx in range(data_num):
        b_mol = data_cover[idx]["mol"]
        b_ig = ig[idx]
        symbol = data_cover[idx]["unimol_symbol"].numpy().tolist()
        b_symbol = list(atom.GetAtomicNum() for atom in b_mol.GetAtoms())
        assert symbol == b_symbol, "Symbol != b_Symbol"
        assert len(b_ig) == len(b_symbol), "b_big != b_Symbol"

        atoms = b_mol.GetAtoms()
        for i in range(len(atoms)):
            atom = atoms[i]
            score = b_ig[i].tolist()
            atom.SetProp("ig_t1", str(score[0]))
            atom.SetProp("ig_t2", str(score[1]))
            atom.SetProp("ig_t3", str(score[2]))
            atom.SetProp("ig_t4", str(score[3]))
        record_mol.append(b_mol)
    return record_mol


def results_extract(pred):
    processed_data = []
    for itm in pred:
        itm_valid = itm
        m1 = itm_valid[:4]
        m2 = itm_valid[4:8]
        m3 = itm_valid[8:12]
        m4 = itm_valid[12:16]
        m5 = itm_valid[16:20]
        combine_results = np.array([m1, m2, m3, m4, m5])
        combine_results = np.mean(combine_results, axis=0)
        processed_data.append(combine_results.tolist())
    return processed_data


def test(test_data, model, cal_ig=None):
    model.eval()
    test_dataloader, protein = test_data.values()
    all_ig = []
    all_pred = []
    for idx, i in enumerate(test_dataloader):
        input_data = {"molecule": i, "protein": protein}
        model_return = model.predict(input_data)
        all_pred += model_return["pred"].detach().cpu().numpy().tolist()
        if cal_ig:
            ig_predict = model.ig_predict(input_data)
            all_ig += ig_predict

    return all_pred, all_ig


def test_for_var(test_data, model, cal_ig=None):
    model.eval()

    def enable_dropout(m):
        if type(m) == torch.nn.Dropout:
            m.train()

    model.apply(enable_dropout)

    test_dataloader, protein = test_data.values()
    all_ig = []
    all_pred = []

    for idx, i in enumerate(test_dataloader):
        input_data = {"molecule": i, "protein": protein}
        model_return = model.predict(input_data)

        all_pred += model_return["pred"].detach().cpu().numpy().tolist()
        if cal_ig:
            model.eval()
            ig_predict = model.ig_predict(input_data)
            all_ig += ig_predict
            model.apply(enable_dropout)
    return all_pred, all_ig


def get_domain(smiles, prob_list):
    similarity = get_similarity_all_task(smiles)
    similarity_df = pd.DataFrame(similarity.T, columns=['sim18', 'sim19', 'sim20', 'sim21', "max18", "max19", "max20","max21"])

    task_prob_list = np.array(prob_list)
    valid_task_prob_mean = np.mean(task_prob_list, axis=0)
    valid_task_prob_var = np.var(task_prob_list, axis=0)
    valid_out_smiles = np.array(smiles)

    valid_task_prob_mean_df = pd.DataFrame(valid_task_prob_mean, columns=['mean18', 'mean19', 'mean20', 'mean21'])
    valid_task_prob_var_df = pd.DataFrame(valid_task_prob_var, columns=['var18', 'var19', 'var20', 'var21'])

    valid_df = pd.DataFrame({
        'smiles': valid_out_smiles,
    })
    valid_df = pd.concat([valid_df, valid_task_prob_mean_df, valid_task_prob_var_df, similarity_df], axis=1)

    domain_results = calculate_domain_levels(valid_df)
    return domain_results


rg = 11


def metric_topk(true, y_scores, percents=tuple(range(1, rg))):
    y_true = np.asarray(true).astype(int)
    y_scores = np.asarray(y_scores)

    assert y_true.ndim == 1 and y_scores.ndim == 1 and y_true.size == y_scores.size
    N = y_true.size
    P = int(y_true.sum())
    pi = np.mean(y_true) if np.any(y_true) else 1e-12

    order = np.argsort(-y_scores, kind="mergesort")
    y_sorted = y_true[order]

    tp_cum = np.cumsum(y_sorted)
    ks = np.arange(1, N + 1)

    precision_at_k = tp_cum / ks
    recall_at_k = (tp_cum / max(P, 1)) if P > 0 else np.zeros_like(tp_cum)
    lift_at_k = precision_at_k / pi

    out = {}
    for r in percents:
        k = max(1, int(np.floor(r / 100.0 * N)))
        out[f"Precision@{r}%"] = float(precision_at_k[k - 1])
        out[f"Recall@{r}%"] = float(recall_at_k[k - 1])
        out[f"Lift@{r}%"] = float(lift_at_k[k - 1])
    return out


def metric(prob, label):
    label = np.array(label)
    label[label == -1] = 0
    out = metric_topk(label, prob)
    aur, aul, aup = calculate_auc_from_topk(out)
    print(aur, aul, aup)


def get_similarity_all_task(smiles):
    train_positive_id18 = pd.read_csv(f'data/domain/task18_train_positive.csv')["smiles"].values.tolist()
    train_positive_id19 = pd.read_csv(f'data/domain/task19_train_positive.csv')["smiles"].values.tolist()
    train_positive_id20 = pd.read_csv(f'data/domain/task20_train_positive.csv')["smiles"].values.tolist()
    train_positive_id21 = pd.read_csv(f'data/domain/task21_train_positive.csv')["smiles"].values.tolist()

    valid_similarity18 = cal_ecfp4_topk_similarity(smiles, train_positive_id18)
    valid_similarity19 = cal_ecfp4_topk_similarity(smiles, train_positive_id19)
    valid_similarity20 = cal_ecfp4_topk_similarity(smiles, train_positive_id20)
    valid_similarity21 = cal_ecfp4_topk_similarity(smiles, train_positive_id21)

    top18 = cal_ecfp4_topk_similarity(smiles, train_positive_id18, k=1)
    top19 = cal_ecfp4_topk_similarity(smiles, train_positive_id19, k=1)
    top20 = cal_ecfp4_topk_similarity(smiles, train_positive_id20, k=1)
    top21 = cal_ecfp4_topk_similarity(smiles, train_positive_id21, k=1)

    return np.array([valid_similarity18, valid_similarity19, valid_similarity20, valid_similarity21, top18, top19, top20, top21])


def cal_ecfp4_topk_similarity(smiles1, smiles2, k=3):
    mols1 = [Chem.MolFromSmiles(s) for s in smiles1]
    mols2 = [Chem.MolFromSmiles(s) for s in smiles2]

    fps1 = [fp_gen.GetFingerprint(m) for m in mols1]
    fps2 = [fp_gen.GetFingerprint(m) for m in mols2]

    actual_k = min(k, len(fps2))

    results = []

    for f1 in fps1:
        sims = DataStructs.BulkTanimotoSimilarity(f1, fps2)
        top_k_sims = sorted(sims, reverse=True)[:actual_k]
        mean_sim = sum(top_k_sims) / actual_k
        results.append(mean_sim)

    return results


def calculate_auc_from_topk(topk_dict):
    percents = list(range(1, rg))
    x = np.array(percents) / 100.0

    recall_values = [topk_dict[f"Recall@{r}%"] for r in percents]
    lift_values = [topk_dict[f"Lift@{r}%"] for r in percents]
    precision_values = [topk_dict[f"Precision@{r}%"] for r in percents]

    aur = auc(x, recall_values) / (x[-1] - x[0])
    aul = auc(x, lift_values) / (x[-1] - x[0])
    aup = auc(x, precision_values) / (x[-1] - x[0])

    return aur, aul, aup


def generate_interpretability_svg(mol, weights, width=500, height=500, colormap_name='bwr'):
    weights = np.array(weights)
    max_abs = np.max(np.abs(weights))
    if max_abs > 1e-9:
        norm_weights = (weights / max_abs).tolist()
    else:
        norm_weights = weights.tolist()

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)

    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.addStereoAnnotation = True

    cmap = cm.get_cmap(colormap_name)

    SimilarityMaps.GetSimilarityMapFromWeights(
        mol,
        norm_weights,
        draw2d=drawer,
        colorMap=cmap,
        contourLines=10,
        alpha=0.4,
        sigma=0.2,
    )

    drawer.FinishDrawing()
    svg_text = drawer.GetDrawingText()

    if '<?xml' in svg_text:
        svg_text = svg_text.split('?>')[-1]
    return svg_text


def generate_interprssetability_svg(mol, weights, width=500, height=500, colormap_name='bwr'):
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)

    opts = drawer.drawOptions()
    opts.clearBackground = False

    cmap = cm.get_cmap(colormap_name)

    SimilarityMaps.GetSimilarityMapFromWeights(
        mol,
        weights,
        draw2d=drawer,
        colorMap=cmap,
        contourLines=12,
        alpha=0.5,
        sigma=0.3
    )

    drawer.FinishDrawing()
    svg_text = drawer.GetDrawingText()

    svg_text = svg_text.replace('<?xml version="1.0" encoding="iso-8859-1"?>', '')

    return svg_text


def probs_to_preds_numpy(probs):
    thresholds = np.array([0.963957586717605,
                           0.98577974205017,
                           0.455895565620064,
                           0.903195939058065,
                           ])

    preds = (probs >= thresholds).astype(int)

    return preds


def get_attention_svg(mol, task_id=4):
    all_plot_record = []
    for s_mol in mol:
        task_list = []
        all_atom = s_mol.GetAtoms()
        for task in range(1, task_id + 1):
            all_weight = []
            for atom in all_atom:
                all_weight.append(float(atom.GetProp(f"ig_t{task}")))

            svg = generate_interpretability_svg(s_mol, all_weight)
            svg = to_base64(svg)
            task_list.append(svg)
        all_plot_record.append(task_list)
    return np.array(all_plot_record)


def to_base64(svg_string: str) -> str:
    if not svg_string:
        return ""

    svg_bytes = svg_string.encode('utf-8')
    b64_bytes = base64.b64encode(svg_bytes)
    b64_str = b64_bytes.decode('utf-8')

    return f"data:image/svg+xml;base64,{b64_str}"


if __name__ == "__main__":
    input_smiles = [r"CSCc1oc(cc1)c2cccnc2", "c1ccccc1", "COc1cc(CNC(=O)CCCC/C=C/C(C)C)ccc1O"]

    pred, key_id, mol = main(input_smiles, config_path="predict_config.yaml")

    print("Predictions:", pred)
    print("Keys:", key_id)
