# aq_edge/utils/mlflow_handler.py
import mlflow
import mlflow.pytorch
import torch
import os
import tempfile
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class MLflowHandler:
    """Handler for MLflow experiment tracking and artifact logging."""

    def __init__(self, experiment_name: str = "air_quality_model", run_name: Optional[str] = None):
        """
        Initialize MLflow handler.

        Args:
            experiment_name (str): Name of the MLflow experiment
            run_name (str, optional): Name for the current run
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.run_started = False

    def start_run(self, config: Dict[str, Any]) -> None:
        """Start MLflow run and log configuration parameters."""
        try:
            mlflow.set_experiment(self.experiment_name)
            mlflow.start_run(run_name=self.run_name)
            self.run_started = True

            # Log all configuration parameters
            self._log_config_params(config)
            print(f"MLflow run started: {mlflow.active_run().info.run_id}")

        except Exception as e:
            print(f"Failed to start MLflow run: {e}")
            self.run_started = False

    def _log_config_params(self, config: Dict[str, Any], prefix: str = "") -> None:
        """Recursively log configuration parameters."""
        for key, value in config.items():
            param_name = f"{prefix}{key}" if prefix else key

            if isinstance(value, dict):
                self._log_config_params(value, f"{param_name}.")
            else:
                try:
                    mlflow.log_param(param_name, value)
                except Exception as e:
                    print(f"Failed to log parameter {param_name}: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to MLflow."""
        if not self.run_started:
            return

        try:
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value, step=step)
        except Exception as e:
            print(f"Failed to log metrics: {e}")

    def log_model(self, model: torch.nn.Module, model_name: str = "pytorch_model") -> None:
        """Log PyTorch model to MLflow."""
        if not self.run_started:
            return

        try:
            mlflow.pytorch.log_model(model, model_name)
            print(f"Model logged as '{model_name}'")
        except Exception as e:
            print(f"Failed to log model: {e}")

    def log_plot(self, figure: plt.Figure, plot_name: str) -> None:
        """Log matplotlib plot to MLflow."""
        if not self.run_started:
            return

        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                figure.savefig(tmp_file.name, dpi=300, bbox_inches='tight')
                mlflow.log_artifact(tmp_file.name, f"plots/{plot_name}.png")
                os.unlink(tmp_file.name)
            print(f"Plot logged: {plot_name}")
        except Exception as e:
            print(f"Failed to log plot {plot_name}: {e}")

    def log_artifact(self, file_path: str, artifact_path: Optional[str] = None) -> None:
        """Log file as artifact to MLflow."""
        if not self.run_started:
            return

        try:
            mlflow.log_artifact(file_path, artifact_path)
            print(f"Artifact logged: {file_path}")
        except Exception as e:
            print(f"Failed to log artifact {file_path}: {e}")

    def log_predictions(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """Log predictions and targets as artifacts."""
        if not self.run_started:
            return

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Save predictions and targets
                pred_path = os.path.join(tmp_dir, "predictions.npy")
                target_path = os.path.join(tmp_dir, "targets.npy")

                np.save(pred_path, predictions)
                np.save(target_path, targets)

                # Log as artifacts
                mlflow.log_artifact(pred_path, "predictions")
                mlflow.log_artifact(target_path, "predictions")

            print("Predictions and targets logged")
        except Exception as e:
            print(f"Failed to log predictions: {e}")

    def end_run(self) -> None:
        """End MLflow run."""
        if self.run_started:
            try:
                mlflow.end_run()
                print("MLflow run ended successfully")
            except Exception as e:
                print(f"Failed to end MLflow run: {e}")
            finally:
                self.run_started = False