"""
Configuration loader for prediction
Loads prediction settings from YAML config file
"""

import yaml
import torch
from pathlib import Path
from typing import Dict, Any


def load_predict_config(config_path: str = "predict_config.yaml") -> Dict[str, Any]:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def get_device_from_config(config: Dict[str, Any] = None, config_path: str = None) -> str:
    if config is None:
        if config_path is None:
            config_path = "predict_config.yaml"
        config = load_predict_config(config_path)

    device_config = config.get('model', {}).get('device', 'cuda')

    if device_config == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_config

    return device


def get_model_params_from_config(config: Dict[str, Any] = None, config_path: str = None) -> Dict[str, Any]:
    if config is None:
        if config_path is None:
            config_path = "predict_config.yaml"
        config = load_predict_config(config_path)

    model_config = config.get('model', {})
    loss_config = config.get('loss', {})
    dropout_config = config.get('dropout', {})
    descriptor_config = config.get('descriptor', {})

    params = {
        'hidden_channels': model_config.get('hidden_channels', 256),
        'out_channels': model_config.get('out_channels', 256),
        'protein_input_dim': model_config.get('protein_input_dim', 1152),
        'sub_model_num': model_config.get('sub_model_num', 5),
        'task_num': model_config.get('task_num', 4),
        'device': get_device_from_config(config),
        'prior': model_config.get('prior', None),
        'contrastive_tau': loss_config.get('contrastive_tau', 0.1),
        'contrastive_weight': loss_config.get('contrastive_weight', 0.1),
        'gate_dropout': dropout_config.get('gate', 0.2),
        'fusion_dropout': dropout_config.get('fusion', 0.3),
        'des_dropout': dropout_config.get('descriptor', 0.1),
    }

    return params


def get_prediction_params_from_config(config: Dict[str, Any] = None, config_path: str = None) -> Dict[str, Any]:
    if config is None:
        if config_path is None:
            config_path = "predict_config.yaml"
        config = load_predict_config(config_path)

    pred_config = config.get('prediction', {})

    params = {
        'batch_size': pred_config.get('batch_size', 128),
        'model_checkpoint': pred_config.get('model_checkpoint', 'model_results/final_renamed_model.pth'),
        'shuffle': pred_config.get('shuffle', False),
        'num_workers': pred_config.get('num_workers', 0),
    }

    return params


def create_model_for_prediction(config_path: str = "predict_config.yaml"):
    from backbone import ULCYP

    config = load_predict_config(config_path)
    model_params = get_model_params_from_config(config)

    model = ULCYP(**model_params)

    return model, config
