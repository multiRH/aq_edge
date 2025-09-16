import os
from dotenv import load_dotenv

# Load environment variables immediately and silently
load_dotenv(override=True)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import traceback
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

# Import all modules
from aq_edge.utils.logging import LoggerHandler
from aq_edge.utils.config import ConfigHandler
from aq_edge.datautils.preprocessing import prepare_data
from aq_edge.modelzoo.lstm import BaseLSTM, AttentionLSTM
from aq_edge.modelzoo.model_factory import EarlyStopping
from aq_edge.evaluation.metrics import calculate_horizon_metrics
from aq_edge.utils.visualization import (
    plot_horizon_predictions, plot_horizon_metrics, plot_loss_curves, plot_r2_curves,
    plot_forecast_vs_ground_truth
)
from aq_edge.utils.mlflow_handler import MLflowHandler, generate_custom_run_name

if __name__ == "__main__":
# def main():
    # Initialize logger first (this creates the consolidated log file)
    logger = LoggerHandler(__name__)
    logger.info("Starting air quality prediction pipeline")

    try:
        # Load configuration
        config = ConfigHandler()
        logger.info("Configuration loaded successfully")

        # Set random seeds for reproducibility
        seed = config.get('training.random_seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        logger.info(f"Random seed set to: {seed}")

        # MLflow integration
        mlflow_enabled = config.get('mlflow.enabled', True)
        custom_run_name = generate_custom_run_name()  # Generate custom run name
        mlflow_handler = MLflowHandler(
            experiment_name=f"air_quality_{config.get('data.station', 'default')}",
            run_name=custom_run_name,
            enabled=mlflow_enabled
        )

        logger.info(f"Starting MLflow run with name: {custom_run_name}")

        # Start MLflow run
        if mlflow_handler.start_run(config.config):
            logger.info("MLflow run started successfully")
        else:
            logger.warning("MLflow run failed to start, continuing without tracking")

        logger.info("="*70)
        logger.info("AIR QUALITY MODEL TRAINING AND EVALUATION")
        logger.info("="*70)

        # -----------------------------
        # 1. Load data
        # -----------------------------
        station = config.get('data.station')
        data_path = config.get('data.data_path').format(station=station)
        drop_columns = config.get('data.drop_columns', [])

        logger.info(f"Loading air quality data for station: {station}")
        logger.info(f"Data path: {data_path}")

        df = pd.read_parquet(data_path)
        logger.info(f"Original data shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")

        # Drop specified columns
        data = df.drop(columns=drop_columns)
        logger.info(f"Dropped columns: {drop_columns}")

        # Define features and target for modeling
        features = config.get('data.features')
        target = config.get('data.target')

        logger.info(f"Features for station {station}: {features}")
        logger.info(f"Target: {target}")

        # MLflow: Log data info
        mlflow_handler.log_metrics({
            "data_shape_rows": df.shape[0],
            "data_shape_cols": df.shape[1],
            "num_features": len(features)
        })

        # -----------------------------
        # 2. Prepare data
        # -----------------------------
        logger.info("Starting data preparation...")
        preprocessing_config = config.get_section('preprocessing')

        # Use the new prepare_data function
        prepared_data = prepare_data(
            data=data,
            features=features,
            target=target,
            train_end_timestamp=preprocessing_config['train_end_timestamp'],
            val_end_timestamp=preprocessing_config['val_end_timestamp'],
            input_sequence_length=preprocessing_config['input_sequence_length'],
            output_sequence_length=preprocessing_config['output_sequence_length'],
            batch_size=config.get('training.batch_size')
        )

        train_dataloader = prepared_data['train_dataloader']
        validation_dataloader = prepared_data['validation_dataloader']
        test_dataloader = prepared_data['test_dataloader']

        logger.info("Data preparation completed successfully!")
        logger.info(f"Number of training sequences: {len(prepared_data['train_dataset'])}")
        logger.info(f"Number of validation sequences: {len(prepared_data['validation_dataset'])}")
        logger.info(f"Number of test sequences: {len(prepared_data['test_dataset'])}")

        logger.info(f"CO2 target statistics:")
        logger.info(f"Min: {data[target].min():.2f}")
        logger.info(f"Max: {data[target].max():.2f}")
        logger.info(f"Mean: {data[target].mean():.2f}")
        logger.info(f"Std: {data[target].std():.2f}")

        # MLflow: Log dataset sizes
        mlflow_handler.log_metrics({
            "train_sequences": len(prepared_data['train_dataset']),
            "val_sequences": len(prepared_data['validation_dataset']),
            "test_sequences": len(prepared_data['test_dataset'])
        })

        # -----------------------------
        # 3. Initialize model
        # -----------------------------
        logger.info("Initializing model...")
        use_cuda = config.get('device.use_cuda', True)
        device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        logger.info(f"Using device: {device}")

        model_type = config.get('model.type', 'AttentionLSTM')
        input_size = len(features)
        output_size = preprocessing_config['output_sequence_length']

        if model_type == 'AttentionLSTM':
            model = AttentionLSTM(input_size=input_size, output_size=output_size).to(device)
        else:
            model = BaseLSTM(input_size=input_size, output_size=output_size).to(device)

        logger.info(f"Model type: {model_type}")
        logger.info(f"Input size: {input_size}")
        logger.info(f"Output size: {output_size}")

        # Initialize optimizer and criterion
        training_config = config.get_section('training')
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
        )
        criterion = nn.MSELoss()

        model_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {model_params:,}")
        logger.info(f"Learning rate: {training_config['learning_rate']}")
        logger.info(f"Weight decay: {training_config['weight_decay']}")

        # MLflow: Log model info
        mlflow_handler.log_metrics({
            "model_parameters": model_params,
            "input_size": input_size,
            "output_size": output_size
        })

        # -----------------------------
        # 4. Check MLflow Model Registry
        # -----------------------------
        model_name = f"{config.get('data.station')}_{model_type}_model"
        model_version = None

        if mlflow_enabled:
            logger.info(f"Checking MLflow model registry for: {model_name}")
            try:
                # Try to get and load existing model
                model_version = mlflow_handler.get_latest_model_version(model_name)

                if model_version:
                    logger.info(f"Found model {model_name} version {model_version}")
                    model_uri = f"models:/{model_name}/{model_version}"

                    loaded_model = mlflow_handler.load_model(model_uri)
                    if loaded_model:
                        model = loaded_model.to(device)
                        logger.info("Model loaded from registry, skipping training")
                        skip_training = True
                    else:
                        logger.warning("Failed to load model, proceeding with training")
                        skip_training = False
                else:
                    logger.info("No existing model found, proceeding with training")
                    skip_training = False

            except Exception as e:
                logger.error(f"Registry check failed: {str(e)}")
                logger.info("Proceeding with training due to registry issues")
                skip_training = False
        else:
            skip_training = False

        # -----------------------------
        # 4. Training Loop
        # -----------------------------

        if not skip_training:

            logger.info("Starting training...")
            num_epochs = training_config['num_epochs']
            patience = training_config['patience']
            early_stopping = EarlyStopping(patience=patience)
            train_losses, val_losses = [], []

            logger.info(f"Number of epochs: {num_epochs}")
            logger.info(f"Early stopping patience: {patience}")

            # Initialize lists for R² tracking
            train_r2_scores, val_r2_scores = [], []

            for epoch in range(num_epochs):
                model.train()
                batch_losses = []
                train_predictions_epoch = []
                train_targets_epoch = []

                for batch_idx, (xb, yb) in enumerate(train_dataloader):
                    xb, yb = xb.to(device), yb.to(device)

                    if epoch == 0 and batch_idx == 0:
                        logger.info(f"First batch input shape: {xb.shape}")
                        logger.info(f"First batch target shape: {yb.shape}")

                    optimizer.zero_grad()
                    out = model(xb)

                    if yb.dim() > 2:
                        yb = yb.squeeze(-1)

                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(loss.item())

                    # Collect predictions and targets for R² calculation
                    train_predictions_epoch.append(out.detach().cpu().numpy())
                    train_targets_epoch.append(yb.cpu().numpy())

                train_loss = np.mean(batch_losses)
                train_losses.append(train_loss)

                # Calculate training R²
                train_pred_flat = np.concatenate(train_predictions_epoch).flatten()
                train_true_flat = np.concatenate(train_targets_epoch).flatten()
                train_r2 = r2_score(train_true_flat, train_pred_flat)
                train_r2_scores.append(train_r2)

                # Validation
                model.eval()
                val_batch_losses = []
                val_predictions_epoch = []
                val_targets_epoch = []

                with torch.no_grad():
                    for xb, yb in validation_dataloader:
                        xb, yb = xb.to(device), yb.to(device)
                        out = model(xb)

                        if yb.dim() > 2:
                            yb = yb.squeeze(-1)

                        loss = criterion(out, yb)
                        val_batch_losses.append(loss.item())

                        # Collect validation predictions and targets for R²
                        val_predictions_epoch.append(out.cpu().numpy())
                        val_targets_epoch.append(yb.cpu().numpy())

                val_loss = np.mean(val_batch_losses)
                val_losses.append(val_loss)

                # Calculate validation R²
                val_pred_flat = np.concatenate(val_predictions_epoch).flatten()
                val_true_flat = np.concatenate(val_targets_epoch).flatten()
                val_r2 = r2_score(val_true_flat, val_pred_flat)
                val_r2_scores.append(val_r2)

                # Log every 10 epochs or last epoch
                if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1:
                    logger.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}")

                # MLflow: Log metrics every epoch
                mlflow_handler.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_r2": train_r2,
                    "val_r2": val_r2
                }, step=epoch)

                # Check early stopping
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

            logger.info("Training completed!")

            # Register the new model
            logger.info("Registering trained model to MLflow registry...")
            mlflow_handler.register_model(model, model_name, "Production")

        else:
            logger.info("Skipping training - using model from registry")
            # Initialize empty lists for visualization compatibility
            train_losses, val_losses = [], []
            train_r2_scores, val_r2_scores = [], []

        # -----------------------------
        # 5. Test Evaluation
        # -----------------------------
        logger.info("Starting test evaluation...")
        model.eval()
        test_batch_losses = []
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for xb, yb in test_dataloader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)

                if yb.dim() > 2:
                    yb = yb.squeeze(-1)

                loss = criterion(out, yb)
                test_batch_losses.append(loss.item())

                all_predictions.append(out.cpu().numpy())
                all_targets.append(yb.cpu().numpy())

        test_loss = np.mean(test_batch_losses)
        logger.info(f"Test Loss: {test_loss:.6f}")

        # Convert to numpy arrays for analysis
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        logger.info(f"Test predictions shape: {predictions.shape}")
        logger.info(f"Test targets shape: {targets.shape}")

        # MLflow: Log test loss and predictions
        mlflow_handler.log_metrics({"test_loss": test_loss})
        mlflow_handler.log_predictions(predictions, targets)

        # -----------------------------
        # 6. Visualization
        # -----------------------------
        logger.info("Creating visualizations...")
        viz_config = config.get_section('visualization')
        plot_dir = viz_config.get('plot_dir', 'plots')
        os.makedirs(plot_dir, exist_ok=True)

        # Only create loss and R² plots if we actually trained the model
        if not skip_training and train_losses and val_losses:

            # Create loss curves plot
            logger.info("Creating loss curves plot...")
            fig_loss = plot_loss_curves(
                train_losses=train_losses,
                val_losses=val_losses,
                figsize=(8, 6)
            )

            if viz_config.get('save_plots', False):
                plot_path = f"{plot_dir}/loss_curves.png"
                fig_loss.savefig(plot_path, dpi=150, bbox_inches='tight')
                logger.info(f"Loss curves plot saved to {plot_path}")

            # MLflow: Log loss curves plot
            mlflow_handler.log_plot(fig_loss, "loss_curves")
            plt.show()

            # If you have R² scores, create R² curves plot
            if 'train_r2_scores' in locals() and 'val_r2_scores' in locals():
                logger.info("Creating R² curves plot...")
                fig_r2 = plot_r2_curves(
                    train_r2_scores=train_r2_scores,
                    val_r2_scores=val_r2_scores,
                    figsize=(8, 6)
                )

                if viz_config.get('save_plots', False):
                    r2_plot_path = f"{plot_dir}/r2_curves.png"
                    fig_r2.savefig(r2_plot_path, dpi=150, bbox_inches='tight')
                    logger.info(f"R² curves plot saved to {r2_plot_path}")

                # MLflow: Log R² curves plot
                mlflow_handler.log_plot(fig_r2, "r2_curves")
                plt.show()
        else:
            logger.info("Skipping loss/R² plots - model loaded from registry or no training data")

        # Create forecast vs ground truth plot
        if 'predictions' in locals() and 'targets' in locals():
            logger.info("Creating forecast vs ground truth plot...")

            # Extract test timestamps and targets from prepared_data
            test_timestamps = prepared_data['test_timestamps']
            y_test_seq = prepared_data['y_test_seq']

            fig_forecast = plot_forecast_vs_ground_truth(
                timestamps=test_timestamps,
                ground_truth=targets,
                forecasts=predictions,
                horizons=[1, 3, 9, 12],
                figsize=(15, 10)
            )

            if viz_config.get('save_plots', False):
                forecast_plot_path = f"{plot_dir}/forecast_vs_ground_truth.png"
                fig_forecast.savefig(forecast_plot_path, dpi=150, bbox_inches='tight')
                logger.info(f"Forecast vs ground truth plot saved to {forecast_plot_path}")

            # MLflow: Log forecast plot
            mlflow_handler.log_plot(fig_forecast, "forecast_vs_ground_truth")
            plt.show()

        # Calculate and log horizon metrics
        logger.info("Calculating metrics per horizon...")
        horizon_metrics = calculate_horizon_metrics(predictions, targets)

        # Print and log summary statistics
        logger.info("Horizon Metrics Summary:")
        summary_metrics = {}
        for metric, values in horizon_metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            logger.info(f"{metric}: Mean={mean_val:.4f}, Std={std_val:.4f}")

            # MLflow: Log summary metrics
            summary_metrics[f"{metric}_mean"] = mean_val
            summary_metrics[f"{metric}_std"] = std_val

        mlflow_handler.log_metrics(summary_metrics)

        # Plot metrics per horizon
        logger.info("Creating horizon metrics plot...")
        fig_metrics = plot_horizon_metrics(horizon_metrics)
        mlflow_handler.log_plot(fig_metrics, "horizon_metrics")

        # MLflow: Log the trained model
        logger.info("Logging trained model to MLflow...")
        mlflow_handler.log_model(model, "trained_model")

        # MLflow: Log config file as artifact
        if hasattr(config, 'config_path') and os.path.exists(config.config_path):
            mlflow_handler.log_artifact(config.config_path, "config")
            logger.info("Configuration file logged to MLflow")

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

    finally:
        # Log all individual log files to MLflow before ending the run
        if LoggerHandler.log_all_to_mlflow(mlflow_handler):
            logger.info("All log files uploaded to MLflow as artifacts")
        else:
            logger.warning("Some log files failed to upload to MLflow")

        # End MLflow run
        if mlflow_handler.end_run():
            logger.info("MLflow run ended successfully")
        else:
            logger.warning("Failed to end MLflow run properly")

# if __name__ == "__main__":
#     main()