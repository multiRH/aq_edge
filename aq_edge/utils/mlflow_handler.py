import mlflow
import mlflow.pytorch
import os
import pickle
from typing import Dict, Any, Optional, List
import numpy as np
import torch

class MLflowHandler:
    """MLflow handler for experiment tracking"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mlflow_config = config['mlflow']

        # Set tracking URI
        mlflow.set_tracking_uri(self.mlflow_config['tracking_uri'])

        # Set or create experiment
        try:
            experiment_id = mlflow.create_experiment(self.mlflow_config['experiment_name'])
        except mlflow.exceptions.MlflowException:
            experiment = mlflow.get_experiment_by_name(self.mlflow_config['experiment_name'])
            experiment_id = experiment.experiment_id

        mlflow.set_experiment(experiment_id=experiment_id)

    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """Start MLflow run"""
        run_name = run_name or self.mlflow_config.get('run_name', 'solar_forecasting_run')
        return mlflow.start_run(run_name=run_name)

    def log_config(self):
        """Log configuration parameters"""
        # Log model parameters
        for key, value in self.config['model'].items():
            mlflow.log_param(f"model_{key}", value)

        # Log training parameters
        for key, value in self.config['training'].items():
            mlflow.log_param(f"training_{key}", value)

        # Log data parameters
        for key, value in self.config['data'].items():
            mlflow.log_param(f"data_{key}", value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_model(self, model: torch.nn.Module, preprocessor: Any):
        """Log PyTorch model and preprocessor"""
        # Log the PyTorch model
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=f"solar_forecasting_{self.config['model']['type']}"
        )

        # Log preprocessor separately
        preprocessor_path = "preprocessor.pkl"
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(preprocessor, f)
        mlflow.log_artifact(preprocessor_path, "preprocessor")
        os.remove(preprocessor_path)

    def log_artifacts(self, artifact_paths: List[str], artifact_folder: Optional[str] = None):
        """Log multiple artifacts"""
        for path in artifact_paths:
            if os.path.exists(path):
                mlflow.log_artifact(path, artifact_folder)

    def log_data_info(self, data_info: Dict[str, Any]):
        """Log data information"""
        for key, value in data_info.items():
            mlflow.log_param(f"data_info_{key}", value)

    def end_run(self):
        """End MLflow run"""
        mlflow.end_run()