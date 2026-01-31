import os
import math
import numpy as np
import pandas as pd
from typing import Dict, Any

def _to_float(x):
    """Safely convert to float, preserving NaN."""
    if x is None:
        return np.nan
    if isinstance(x, (np.floating, float, int)):
        return float(x)
    # Sometimes it's numpy scalar or convertible string
    try:
        return float(x)
    except Exception:
        return x  # Keep original value (e.g., non-numeric), to avoid losing information

def _flatten_epoch_metrics(epoch: int,
                           phase: str,
                           metrics_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    metrics_dict format:
    {
      'task_0': {'AUC': 0.88, 'PRAUC': 0.87, 'Precision@10%': 0.95, ...},
      'task_1': {...},
      ...
    }
    Returns flattened DataFrame: one row = all metrics of one task in that epoch/phase
    """
    rows = []
    if metrics_dict is None:
        return pd.DataFrame()
    for task_name, m in metrics_dict.items():
        row = {"epoch": epoch, "phase": phase, "task": task_name}
        if isinstance(m, dict):
            for k, v in m.items():
                row[k] = _to_float(v)
        rows.append(row)
    return pd.DataFrame(rows)

def log_epoch_metrics_to_csv(epoch: int,
                             metrics_by_phase: Dict[str, Dict[str, Dict[str, Any]]],
                             csv_path: str):
    """
    Flatten and append metrics of multiple phases (e.g., 'valid', 'test', 'test_concat_pu') in one epoch to CSV.
    - metrics_by_phase: {'valid': metrics_epoch_valid, 'test': metrics_epoch_test, ...}
    - Create with header if csv doesn't exist, otherwise append
    """
    frames = []
    for phase, metrics_dict in metrics_by_phase.items():
        df = _flatten_epoch_metrics(epoch, phase, metrics_dict)
        if not df.empty:
            frames.append(df)

    if not frames:
        print(f"[WARN] epoch {epoch}: No metrics to write.")
        return

    df_all = pd.concat(frames, ignore_index=True)

    # Append write
    if not os.path.exists(csv_path):
        df_all.to_csv(csv_path, index=False)
        print(f"[INFO] Created and wrote: {csv_path}, {len(df_all)} rows")
    else:
        # Align columns (prevent inconsistent metric keys across different epochs/phases)
        old = pd.read_csv(csv_path)
        # Unified column set
        all_cols = list(dict.fromkeys(list(old.columns) + list(df_all.columns)))
        old = old.reindex(columns=all_cols)
        df_all = df_all.reindex(columns=all_cols)
        out = pd.concat([old, df_all], ignore_index=True)
        out.to_csv(csv_path, index=False)
        print(f"[INFO] Appended to: {csv_path}, added {len(df_all)} rows, total {len(out)} rows")