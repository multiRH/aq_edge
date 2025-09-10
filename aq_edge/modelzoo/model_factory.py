import torch
from .lstm import BaseLSTM, AttentionLSTM
from typing import Dict, Any

class ModelFactory:
    """Factory for Model creation"""

    @staticmethod
    def create_model(model_type: str, config: Dict[str, Any]) -> torch.nn.Module:
        """Create a model based on the type and configuration"""
        if model_type.lower() == 'lstm':
            return BaseLSTM(
                input_size=config.get('input_size', 1),
                hidden_size=config.get('hidden_size', 64),
                num_layers=config.get('num_layers', 2),
                output_size=config.get('output_size', 12),
                dropout=config.get('dropout', 0.2)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def get_available_models():
        """Returns available model types"""
        return ['lstm']


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
