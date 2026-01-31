import pandas as pd
import numpy as np


def calculate_domain_levels(df):
    """
    Calculate applicability domain levels based on sim (ECFP similarity)
    and var (Sigma uncertainty) along with task-specific thresholds.

    Threshold source: Based on new statistical table (S1/V1=Loose, S2/V2=Strict)
    Task 18 (1A2), Task 19 (2B6), Task 20 (2C), Task 21 (3A4)
    """

    thresholds_map = {
        18: {
            'e_loose': 0.133425436,
            'e_strict': 0.1939,
            's_loose': 0.0112299074319139,
            's_strict': 0.003428182
        },
        19: {
            'e_loose': 0.144620811,
            'e_strict': 0.205,
            's_loose': 0.00161988149729765,
            's_strict': 0.000273625
        },
        20: {
            'e_loose': 0.138308998,
            'e_strict': 0.192,
            's_loose': 0.0451964703954188,
            's_strict':0.025795047
        },
        21: {
            'e_loose': 0.160116686,
            'e_strict': 0.2238,
            's_loose': 0.0520759368246102,
            's_strict': 0.01713253
        }
    }

    result_df = df.copy()

    for task_id, t_vals in thresholds_map.items():

        sim_col = f'sim{task_id}'
        var_col = f'var{task_id}'
        domain_col = f'domain{task_id}'

        if sim_col not in df.columns or var_col not in df.columns:
            continue

        sim = result_df[sim_col]
        var = result_df[var_col]

        EL = t_vals['e_loose']
        ES = t_vals['e_strict']
        SL = t_vals['s_loose']
        SS = t_vals['s_strict']

        conditions = [
            (sim >= ES) & (var <= SS),
            (sim < EL) & (var <= SL),
            (sim >= EL) & (var > SL),
            (sim < EL) | (var > SL)
        ]

        choices = [5, 3, 2, 1]

        result_df[domain_col] = np.select(conditions, choices, default=4)

    return result_df


if __name__ == "__main__":
    data = {
        'SMILES': ['Mol_HighConf', 'Mol_OOD_Var', 'Mol_OOD_Sim', 'Mol_Mid'],
        'var19': [0.001, 0.05, 0.002, 0.02],
        'sim19': [0.21, 0.21, 0.10, 0.15],
    }
    df_test = pd.DataFrame(data)
    df_out = calculate_domain_levels(df_test)
    print(df_out[['SMILES', 'var19', 'sim19', 'domain19']])