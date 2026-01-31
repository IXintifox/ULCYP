from predict_main import main, metric, get_attention_svg, main_with_var, get_domain, probs_to_preds_numpy
import pandas as pd
import numpy as np
from tools.rate_function import rate_probs
from rdkit import Chem
import os

def run(path, ig=False):
    state, smiles = read_smiles_from_file(path)
    if state == 1:
        smiles = smiles
        pred, valid_idx, mol = main_with_var(smiles, cal_ig=ig)

        domain_df = get_domain(np.array(smiles)[valid_idx], pred)

        domain_column = domain_df.loc[:, ["domain18", "domain19", "domain20", "domain21"]].values
        pred_column = domain_df.loc[:, ['mean18', 'mean19', 'mean20', 'mean21']].values
        max_sim_column = domain_df.loc[:, ['max18', 'max19', 'max20', 'max21']].values
        rate = rate_probs(pred_column)
        task_pred = probs_to_preds_numpy(pred_column)
        full_preds = np.full((len(smiles), 4), 7, dtype=np.int32)
        full_preds_thr = np.full((len(smiles), 4), 7, dtype=np.int32)
        full_svg = np.full((len(smiles), 4), 7,  dtype=object)
        full_max_sim = np.full((len(smiles), 4), 7, dtype=np.float32)
        full_domain = np.full((len(smiles), 4), 5, dtype=np.int32)

        if ig:
            att_svg = get_attention_svg(mol)
            full_svg[np.array(valid_idx), :4] = att_svg

        full_preds[np.array(valid_idx), :4] = rate
        full_domain[np.array(valid_idx), :4] = domain_column
        full_max_sim[np.array(valid_idx), :4] = max_sim_column
        full_preds_thr[np.array(valid_idx), :4] = task_pred


        full_preds = np.concatenate((full_preds, full_svg, full_domain, full_preds_thr, full_max_sim), axis=-1)
        full_preds = full_preds.tolist()
        result_df = pd.DataFrame(
            full_preds,
            columns=['task_1', 'task_2', 'task_3', 'task_4', "svg1", "svg2", "svg3", "svg4", "domain1", "domain2", "domain3", "domain4",
                     "pred1", "pred2", "pred3", "pred4", "max1", "max2", "max3", "max4"],
        )
        return 1, result_df
    else:
        return state, smiles


def read_smiles_from_file(filepath, max_samples=300):
    # 状态2: 文件不存在
    if not os.path.exists(filepath):
        return 2, f"无法找到文件: {filepath}"

    try:
        ext = os.path.splitext(filepath)[1].lower()

        # CSV文件
        if ext == '.csv':
            try:
                df = pd.read_csv(filepath)
            except Exception as e:
                return 3, f"CSV读取错误: {str(e)}"

            # 状态4: 找不到SMILES列
            smile_cols = [col for col in df.columns if 'smile' in col.lower()]
            if not smile_cols:
                return 4, "无法找到SMILES列"

            smiles = df[smile_cols[0]].fillna('').astype(str).tolist()

            # 🔴 新增：检查样本数量
            if len(smiles) > max_samples:
                return 6, f"样本数量超过限制 ({len(smiles)} > {max_samples})"

            return 1, smiles

        # Excel文件
        elif ext in ['.xlsx', '.xls']:
            try:
                df = pd.read_excel(filepath)
            except Exception as e:
                return 3, f"Excel读取错误: {str(e)}"

            # 状态4: 找不到SMILES列
            smile_cols = [col for col in df.columns if 'smile' in col.lower()]
            if not smile_cols:
                return 4, "无法找到SMILES列"

            smiles = df[smile_cols[0]].fillna('').astype(str).tolist()

            # 🔴 新增：检查样本数量
            if len(smiles) > max_samples:
                return 6, f"样本数量超过限制 ({len(smiles)} > {max_samples})"

            return 1, smiles

        # SDF/mol文件
        elif ext in ['.sdf', '.mol', '.mol2']:
            try:
                # 尝试读取分子文件
                if ext == '.mol2':
                    mols = Chem.MolFromMol2File(filepath, removeHs=False)
                    mols = [mols] if mols else []
                else:
                    mols = Chem.SDMolSupplier(filepath)

                smiles = []
                for mol in mols:
                    if mol is not None:
                        try:
                            smi = Chem.MolToSmiles(mol)
                            smiles.append(smi)
                            # 🔴 新增：在读取过程中检查数量
                            if len(smiles) > max_samples:
                                return 6, f"样本数量超过限制 (> {max_samples})"
                        except:
                            smiles.append('')
                    else:
                        smiles.append('')

                # 如果文件非空但读取不到分子
                if not smiles and os.path.getsize(filepath) > 0:
                    return 3, "SDF/Mol文件无法解析"

                # 🔴 新增：最终检查数量
                if len(smiles) > max_samples:
                    return 6, f"样本数量超过限制 ({len(smiles)} > {max_samples})"

                return 1, smiles

            except Exception as e:
                return 3, f"SDF/Mol读取错误: {str(e)}"

        else:
            return 5, f"不支持的文件格式: {ext}"
    except Exception as e:
        # 状态5: 其他错误
        return 5, f"出现错误: {str(e)}"


if __name__ == '__main__':
    results, state = run(f"test_data.csv")
    print(results)
