import mlflow
import mlflow.pytorch
import torch
import os
import tempfile
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
from dotenv import load_dotenv
import random
from mlflow.tracking import MlflowClient
from aq_edge.utils.logging import LoggerHandler

def generate_custom_run_name() -> str:
    """Generate a unique run name using Aztec deity names

    Returns:
        str: Custom run name in format 'adjective-deity-number'
    """
    nouns = [
        "Quetzalcóatl", "Tezcatlipoca", "Huitzilopochtli", "Tlaloc", "Xochiquetzal",
        "Coyolxauhqui", "Tonatiuh", "Metztli", "Cihuacóatl", "Mictlantecuhtli"
    ]
    adjectives = [
        "wise", "mysterious", "fierce", "stormy", "graceful",
        "rebellious", "brilliant", "mystical", "protective", "grim"
    ]
    return f"{random.choice(adjectives)}-{random.choice(nouns)}-{random.randint(100, 999)}"

class MLflowHandler:
    """Handler for MLflow experiment tracking and artifact logging with safe execution."""

    def __init__(self, experiment_name: str = "air_quality_model", run_name: Optional[str] = None,
                 enabled: bool = True):
        """
        Initialize MLflow handler.

        Args:
            experiment_name (str): Name of the MLflow experiment
            run_name (str, optional): Name for the current run
            enabled (bool): Whether MLflow tracking is enabled
        """
        self.logger = LoggerHandler(__name__)
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.run_started = False
        self.enabled = enabled
        self.active_run = None

        # Load environment variables and setup MLflow
        self._load_environment()
        self._setup_mlflow_tracking()
        self._check_mlflow_availability()

    def _load_environment(self) -> None:
        """Load environment variables from .env file silently."""
        try:
            load_dotenv(override=True)
        except Exception:
            # Silently continue if .env loading fails
            pass

    def _setup_mlflow_tracking(self) -> None:
        """Setup MLflow tracking URI from environment variables."""
        if not self.enabled:
            return

        try:
            # Only use MLFLOW_TRACKING_URI
            tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
        except Exception:
            # Silently continue with default configuration if setup fails
            pass

    def _check_mlflow_availability(self) -> None:
        """Check if MLflow is available and can be used."""
        if not self.enabled:
            return

        try:
            # Test MLflow basic functionality by creating/accessing experiment
            mlflow.set_experiment(self.experiment_name)
        except Exception:
            # Disable MLflow if connection fails
            self.enabled = False

    def _safe_execute(self, func, *args, **kwargs) -> bool:
        """Safely execute MLflow operations with error handling."""
        if not self.enabled:
            return False

        try:
            return func(*args, **kwargs)
        except Exception:
            return False

    def start_run(self, config: Dict[str, Any]) -> bool:
        """Start MLflow run and log configuration parameters."""
        if not self.enabled:
            return False

        def _start():
            mlflow.set_experiment(self.experiment_name)
            self.active_run = mlflow.start_run(run_name=self.run_name)
            self.run_started = True

            # Log all configuration parameters
            self._log_config_params(config)
            return True

        success = self._safe_execute(_start)
        if not success:
            self.run_started = False
            self.enabled = False
            self.active_run = None
        return success

    def _log_config_params(self, config: Dict[str, Any], prefix: str = "") -> None:
        """Recursively log configuration parameters."""
        for key, value in config.items():
            param_name = f"{prefix}{key}" if prefix else key

            # Limit parameter name length (MLflow has limits)
            if len(param_name) > 250:
                param_name = param_name[:250]

            if isinstance(value, dict):
                self._log_config_params(value, f"{param_name}.")
            else:
                # Convert value to string and limit length
                str_value = str(value)
                if len(str_value) > 500:
                    str_value = str_value[:500] + "..."

                try:
                    mlflow.log_param(param_name, str_value)
                except Exception:
                    pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> bool:
        """Log metrics to MLflow."""
        if not self.enabled or not self.run_started:
            return False

        def _log():
            for metric_name, value in metrics.items():
                # Ensure value is numeric
                if not isinstance(value, (int, float, np.integer, np.floating)):
                    continue

                # Handle NaN/inf values
                if np.isnan(value) or np.isinf(value):
                    continue

                mlflow.log_metric(metric_name, float(value), step=step)
            return True

        return self._safe_execute(_log)

    def log_tags(self, tags: Dict[str, str]):
        """Log tags to MLflow run"""
        if not self.enabled or not self.active_run:
            return

        try:
            for key, value in tags.items():
                mlflow.set_tag(key, str(value))
            self.logger.info(f"Logged {len(tags)} tags to MLflow")
        except Exception as e:
            self.logger.error(f"Failed to log tags: {str(e)}")

    def log_model(self, model: torch.nn.Module, model_name: str = "pytorch_model") -> bool:
        """Log PyTorch model to MLflow using the current API."""
        if not self.enabled or not self.run_started:
            return False

        def _log():
            # Use the updated MLflow API that doesn't trigger deprecation warnings
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=model_name,
                pip_requirements=None,
                extra_pip_requirements=None,
                conda_env=None,
                code_paths=None,
                signature=None,
                input_example=None,
                await_registration_for=None,
                registered_model_name=None,
                metadata=None
            )
            return True

        return self._safe_execute(_log)

    def log_plot(self, figure: plt.Figure, plot_name: str) -> bool:
        """Log matplotlib plot to MLflow."""
        if not self.enabled or not self.run_started:
            return False

        def _log():
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                try:
                    figure.savefig(tmp_file.name, dpi=300, bbox_inches='tight')
                    mlflow.log_artifact(tmp_file.name, f"plots/{plot_name}.png")
                    return True
                finally:
                    # Ensure temp file is always deleted
                    if os.path.exists(tmp_file.name):
                        os.unlink(tmp_file.name)

        return self._safe_execute(_log)

    def log_artifact(self, file_path: str, artifact_path: Optional[str] = None) -> bool:
        """Log file as artifact to MLflow."""
        if not self.enabled or not self.run_started:
            return False

        if not os.path.exists(file_path):
            return False

        def _log():
            mlflow.log_artifact(file_path, artifact_path)
            return True

        return self._safe_execute(_log)

    def log_predictions(self, predictions: np.ndarray, targets: np.ndarray) -> bool:
        """Log predictions and targets as artifacts."""
        if not self.enabled or not self.run_started:
            return False

        def _log():
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Save predictions and targets
                pred_path = os.path.join(tmp_dir, "predictions.npy")
                target_path = os.path.join(tmp_dir, "targets.npy")

                np.save(pred_path, predictions)
                np.save(target_path, targets)

                # Log as artifacts
                mlflow.log_artifact(pred_path, "predictions")
                mlflow.log_artifact(target_path, "predictions")

                return True

        return self._safe_execute(_log)

    def end_run(self) -> bool:
        """End MLflow run."""
        if not self.enabled or not self.run_started:
            return False

        def _end():
            mlflow.end_run()
            return True

        success = self._safe_execute(_end)
        self.run_started = False
        self.active_run = None
        return success

    def disable(self) -> None:
        """Disable MLflow tracking."""
        self.enabled = False
        if self.run_started:
            self.end_run()

    def is_enabled(self) -> bool:
        """Check if MLflow tracking is enabled and working."""
        return self.enabled and self.run_started

    def get_latest_model_version(self, model_name):
        """Get the latest version of a registered model"""
        try:
            client = MlflowClient()

            # Get all versions of the model
            versions = client.get_latest_versions(model_name, stages=["Production", "Staging", "None"])

            if versions:
                # Return the latest version number
                latest_version = max(versions, key=lambda x: int(x.version))
                self.logger.info(f"Latest version for {model_name}: {latest_version.version}")
                return latest_version.version
            else:
                self.logger.info(f"No versions found for model {model_name}")
                return None

        except Exception as e:
            self.logger.error(f"Error getting latest model version for {model_name}: {str(e)}")
            # Re-raise the exception instead of silently returning None
            raise e

    def load_model(self, model_uri: str):
        """Load model from MLflow."""
        if not self.enabled:
            return None

        try:
            return mlflow.pytorch.load_model(model_uri)
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    def register_model(self, model, model_name: str, stage: str = "Production", description: str = None):
        """Register model to MLflow Model Registry with optional description"""
        if not self.enabled or not self.active_run:
            return None

        try:
            # Log model to current run first
            mlflow.pytorch.log_model(model, "model")

            # Get the model URI
            run_id = self.active_run.info.run_id
            model_uri = f"runs:/{run_id}/model"

            # Register the model
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                description=description
            )

            # Transition to specified stage
            client = MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage=stage
            )

            self.logger.info(f"Model registered: {model_name} version {model_version.version}")
            return model_version.version

        except Exception as e:
            self.logger.error(f"Failed to register model: {str(e)}")
            return None