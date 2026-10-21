from __future__ import annotations

import inspect
import json
import os
import random
import traceback
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import all modules
from aq_edge.utils.logging import LoggerHandler
from aq_edge.utils.config import ConfigHandler
from aq_edge.datautils.preprocessing import prepare_data
from aq_edge.modelzoo.lstm import BaseLSTM, AttentionLSTM
from aq_edge.modelzoo.model_factory import EarlyStopping
from aq_edge.evaluation.metrics import calculate_horizon_metrics
from aq_edge.utils.visualization import (
    plot_horizon_predictions, plot_horizon_metrics, plot_loss_curves, plot_r2_curves,
    plot_forecast_vs_ground_truth, plot_single_horizon_forecast
)
from aq_edge.utils.mlflow_handler import MLflowHandler, generate_custom_run_name

# ============================================================
# Configuration
# ============================================================

SCALER_OPTIONS = ["robust", "standard", "minmax"]

INPUT_LENGTHS = [12, 24, 48, 96, 144]
OUTPUT_LENGTHS = [1, 6, 12, 24]

BATCH_SIZES = [32, 64, 128]
HIDDEN_SIZES = [32, 64, 96, 128, 256]

DEFAULT_SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Data preparation
# ============================================================

def prepare_trial_data(
    data,
    features,
    target,
    preprocessing_config,
    batch_size,
    input_sequence_length,
    output_sequence_length,
    scaler_name,
):
    """
    Reuse the existing prepare_data() function.

    IMPORTANT:
    prepare_data() must support selecting the scaler. This helper
    detects common parameter names. If your function uses another
    name, add it to scaler_aliases below.

    The scaler must be fitted on training data only.
    """
    signature = inspect.signature(prepare_data)
    supported = signature.parameters

    kwargs = {
        "data": data,
        "features": features,
        "target": target,
        "train_end_timestamp":
            preprocessing_config["train_end_timestamp"],
        "val_end_timestamp":
            preprocessing_config["val_end_timestamp"],
        "input_sequence_length": input_sequence_length,
        "output_sequence_length": output_sequence_length,
        "batch_size": batch_size,
    }

    # Support common names for a scaler-selection argument.
    scaler_aliases = (
        "scaler_type",
        "scaler_name",
        "scaler_method",
        "scaling_method",
        "scaler"
    )

    scaler_parameter = next(
        (name for name in scaler_aliases if name in supported),
        None,
    )

    if scaler_parameter is None:
        raise TypeError(
            "prepare_data() must expose a scaler-selection parameter. "
            "Add scaler_type (robust/standard/minmax) to your "
            "prepare_data() function and fit it on training data only."
        )

    kwargs[scaler_parameter] = scaler_name

    # Avoid passing optional parameters that the existing function
    # does not accept.
    kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in supported
    }

    return prepare_data(**kwargs)


# ============================================================
# Model initialization
# ============================================================

def build_model(params, input_size, output_size, device):
    model_classes = {
        "BaseLSTM": BaseLSTM,
        #"AttentionLSTM": AttentionLSTM,
    }

    model_cls = model_classes[params["model_type"]]

    model = model_cls(
        input_size=input_size,
        output_size=output_size,
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    )

    return model.to(device)


# ============================================================
# Training and evaluation
# ============================================================

def evaluate_model(model, dataloader, device, criterion):
    model.eval()

    losses = []
    predictions = []
    targets = []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)

            outputs = model(xb)

            if yb.dim() > 2:
                yb = yb.squeeze(-1)

            loss = criterion(outputs, yb)
            losses.append(loss.item())

            predictions.append(outputs.detach().cpu().numpy())
            targets.append(yb.detach().cpu().numpy())

    if not predictions:
        raise ValueError("Evaluation dataloader contains no batches.")

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    pred_flat = predictions.reshape(-1)
    true_flat = targets.reshape(-1)

    metrics = {
        "loss": float(np.mean(losses)),
        "r2": float(r2_score(true_flat, pred_flat)),
        "rmse": float(np.sqrt(mean_squared_error(true_flat, pred_flat))),
        "mae": float(mean_absolute_error(true_flat, pred_flat)),
    }

    return metrics


def train_trial(
    trial,
    prepared_data,
    params,
    input_size,
    device,
    logger,
):
    train_loader = prepared_data["train_dataloader"]
    val_loader = prepared_data["validation_dataloader"]

    model = build_model(
        params=params,
        input_size=input_size,
        output_size=params["output_sequence_length"],
        device=device,
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )

    scheduler = None

    if params["use_reduce_on_plateau"]:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=params["scheduler_factor"],
            patience=params["scheduler_patience"],
            min_lr=1e-7,
        )

    early_stopping = EarlyStopping(
        patience=params["early_stopping_patience"]
    )

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(params["num_epochs"]):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            if yb.dim() > 2:
                yb = yb.squeeze(-1)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(xb)
            loss = criterion(outputs, yb)

            loss.backward()

            # Helps stabilize recurrent-network training.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))

        val_metrics = evaluate_model(
            model,
            val_loader,
            device,
            criterion,
        )
        val_loss = val_metrics["loss"]

        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        trial.report(val_metrics["r2"], epoch)

        logger.debug(
            "Trial %s | Epoch %d/%d | train_loss=%.6f | "
            "val_loss=%.6f | val_R2=%.5f",
            trial.number,
            epoch + 1,
            params["num_epochs"],
            train_loss,
            val_loss,
            val_metrics["r2"],
        )

        if trial.should_prune():
            raise optuna.TrialPruned(
                "Trial pruned based on intermediate validation R²."
            )

        early_stopping(val_loss)

        if early_stopping.early_stop:
            logger.info(
                "Trial %s: early stopping at epoch %d",
                trial.number,
                epoch + 1,
            )
            break

    if best_state is None:
        raise optuna.TrialPruned("No valid model checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)

    final_metrics = evaluate_model(
        model,
        val_loader,
        device,
        criterion,
    )

    final_metrics["best_epoch"] = best_epoch

    return model, final_metrics


# ============================================================
# Main Optuna pipeline
# ============================================================

def main():
    logger = LoggerHandler(__name__)
    logger.info("Starting CO2 forecasting hyperparameter optimization")

    mlflow_handler = None

    try:
        config = ConfigHandler()

        seed = config.get("training.random_seed", DEFAULT_SEED)
        set_seed(seed)

        use_cuda = config.get("device.use_cuda", True)
        device = torch.device(
            "cuda"
            if torch.cuda.is_available() and use_cuda
            else "cpu"
        )

        logger.info("Device: %s", device)

        # ----------------------------------------------------
        # 1. Load data and configuration
        # ----------------------------------------------------
        station = config.get("data.station")
        data_path = config.get("data.data_path").format(
            station=station
        )

        df = pd.read_parquet(data_path)

        drop_columns = config.get("data.drop_columns", [])
        data = df.drop(
            columns=drop_columns,
            errors="ignore",
        ).copy()

        features = config.get("data.features")
        target = config.get("data.target")

        missing = [
            col for col in list(features) + [target]
            if col not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Available columns: {data.columns.tolist()}"
            )

        # Keep only modeling columns and remove invalid rows.
        data = data[list(dict.fromkeys(list(features) + [target]))]
        data = data.replace([np.inf, -np.inf], np.nan).dropna()

        preprocessing_config = config.get_section("preprocessing")

        logger.info("Station: %s", station)
        logger.info("Features: %s", features)
        logger.info("Target: %s", target)
        logger.info("Valid data shape: %s", data.shape)

        # ----------------------------------------------------
        # 2. Initialize MLflow tracking
        # ----------------------------------------------------
        mlflow_enabled = config.get("mlflow.enabled", True)
        run_name = f"optuna_{station}_{generate_custom_run_name()}"

        mlflow_handler = MLflowHandler(
            experiment_name=f"air_quality_{station}",
            run_name=run_name,
            enabled=mlflow_enabled,
        )

        mlflow_started = mlflow_handler.start_run(config.config)

        if mlflow_started:
            logger.info("MLflow run started: %s", run_name)
        else:
            logger.warning("Continuing without MLflow tracking")

        # ----------------------------------------------------
        # 3. Configure Optuna
        # ----------------------------------------------------
        n_trials = config.get("optuna.n_trials", 100)
        timeout = config.get("optuna.timeout_seconds", None)

        output_dir = Path(
            config.get("optuna.output_dir", "optuna_results")
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        study_name = f"co2_{station}_base_lstm_test"
        storage_path = (output_dir / f"{study_name}.db").resolve()

        study = optuna.create_study(
            study_name=study_name,
            storage=f"sqlite:///{storage_path.as_posix()}",
            direction="maximize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=10,
            ),
        )

        def objective(trial):
            set_seed(seed + trial.number)

            params = {
                "model_type": trial.suggest_categorical(
                    "model_type",
                    ["BaseLSTM"],#, "AttentionLSTM"],
                ),
                "hidden_size": trial.suggest_categorical(
                    "hidden_size",
                    HIDDEN_SIZES,
                ),
                "num_layers": trial.suggest_int(
                    "num_layers", 1, 3
                ),
                "dropout": trial.suggest_float(
                    "dropout", 0.0, 0.4
                ),
                "batch_size": trial.suggest_categorical(
                    "batch_size", BATCH_SIZES
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    1e-5,
                    3e-3,
                    log=True,
                ),
                "weight_decay": trial.suggest_float(
                    "weight_decay",
                    1e-7,
                    1e-3,
                    log=True,
                ),
                "use_reduce_on_plateau": trial.suggest_categorical(
                    "use_reduce_on_plateau",
                    [True, False],
                ),
                "scheduler_factor": trial.suggest_float(
                    "scheduler_factor", 0.2, 0.7
                ),
                "scheduler_patience": trial.suggest_int(
                    "scheduler_patience", 2, 6
                ),
                "early_stopping_patience": trial.suggest_int(
                    "early_stopping_patience", 5, 12
                ),
                "num_epochs": trial.suggest_int(
                    "num_epochs", 20, 100, step=10
                ),
                "input_sequence_length": trial.suggest_categorical(
                    "input_sequence_length", INPUT_LENGTHS
                ),
                "output_sequence_length": trial.suggest_categorical(
                    "output_sequence_length", OUTPUT_LENGTHS
                ),
                "scaler": trial.suggest_categorical(
                    "scaler", SCALER_OPTIONS
                ),
            }

            logger.info(
                "Trial %d/%d parameters: %s",
                trial.number + 1,
                n_trials,
                json.dumps(params),
            )

            prepared_data = prepare_trial_data(
                data=data,
                features=features,
                target=target,
                preprocessing_config=preprocessing_config,
                batch_size=params["batch_size"],
                input_sequence_length=params[
                    "input_sequence_length"
                ],
                output_sequence_length=params[
                    "output_sequence_length"
                ],
                scaler_name=params["scaler"],
            )
            print("++++++++++{features}+++++++++")
            model, metrics = train_trial(
                trial=trial,
                prepared_data=prepared_data,
                params=params,
                input_size=len(features),
                device=device,
                logger=logger,
            )

            trial.set_user_attr("validation_metrics", metrics)
            trial.set_user_attr("best_epoch", metrics["best_epoch"])

            logger.info(
                "Trial %d finished: val_R2=%.5f, "
                "val_RMSE=%.5f, val_MAE=%.5f, best_epoch=%d",
                trial.number,
                metrics["r2"],
                metrics["rmse"],
                metrics["mae"],
                metrics["best_epoch"],
            )

            # R² is invariant to affine target scaling. The target
            # transform must be affine for this property to hold.
            return metrics["r2"]

        # ----------------------------------------------------
        # 4. Run search
        # ----------------------------------------------------
        logger.info(
            "Starting Optuna search: trials=%d, station=%s",
            n_trials,
            station,
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            gc_after_trial=True,
            show_progress_bar=True,
        )

        # ----------------------------------------------------
        # 5. Export best configuration and all trial results
        # ----------------------------------------------------
        best_trial = study.best_trial

        best_result = {
            "station": station,
            "target": target,
            "features": features,
            "objective": "validation_r2",
            "best_validation_r2": best_trial.value,
            "best_parameters": best_trial.params,
            "best_epoch": best_trial.user_attrs.get("best_epoch"),
            "validation_metrics": best_trial.user_attrs.get(
                "validation_metrics"
            ),
            "trial_number": best_trial.number,
            "seed": seed,
        }

        best_path = output_dir / f"best_{study_name}.json"
        with best_path.open("w", encoding="utf-8") as f:
            json.dump(best_result, f, indent=2, default=str)

        trials_path = output_dir / f"trials_{study_name}.csv"
        study.trials_dataframe().to_csv(
            trials_path,
            index=False,
        )

        logger.info(
            "Best validation R²: %.5f",
            best_trial.value,
        )
        logger.info(
            "Best parameters: %s",
            json.dumps(best_trial.params, indent=2),
        )
        logger.info("Best result saved to %s", best_path)
        logger.info("Trial table saved to %s", trials_path)

        if mlflow_handler is not None and mlflow_started:
            mlflow_handler.log_params(best_trial.params)
            mlflow_handler.log_metrics({
                "best_validation_r2": float(best_trial.value),
                "best_epoch": float(
                    best_trial.user_attrs.get("best_epoch", 0)
                ),
            })
            mlflow_handler.log_artifact(str(best_path), "optuna")
            mlflow_handler.log_artifact(str(trials_path), "optuna")

        logger.info("Optuna hyperparameter search completed.")

    except Exception as e:
        logger.error("Optuna search failed: %s", e)
        logger.error(traceback.format_exc())
        raise

    finally:
        if mlflow_handler is not None:
            try:
                mlflow_handler.end_run()
            except Exception:
                logger.exception("Failed to close MLflow run.")


if __name__ == "__main__":
    main()