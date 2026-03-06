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
import subprocess
import json
import pickle

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
        model_info = {
            "model_parameters": int(model_params),
            "learning_rate": float(training_config.get('learning_rate', 0)),
            "weight_decay": float(training_config.get('weight_decay', 0))
        }
        logger.info("Model summary: %s", json.dumps(model_info))

        # MLflow: Log model info
        mlflow_handler.log_metrics({
            "model_parameters": model_params,
            "input_size": input_size,
            "output_size": output_size
        })

        # Get git commit information (move this before skip_training logic)
        try:
            current_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
            commit_short = current_commit[:8]
        except:
            current_commit = "unknown"
            commit_short = "unknown"

        logger.info(f"Git commit: {commit_short}")

        # -----------------------------
        # 4. Check MLflow Model Registry
        # -----------------------------
        # Generate version-aware model name
        num_epochs = training_config['num_epochs']
        learning_rate = training_config['learning_rate']
        batch_size = training_config['batch_size']

        # Create model name with key hyperparameters
        model_name = f"{config.get('data.station')}_{model_type}_e{num_epochs}_lr{str(learning_rate).replace('.', '')}_b{batch_size}_model"

        logger.info(f"Model name with hyperparameters: {model_name}")
        model_version = None

        if mlflow_enabled:
            logger.info(f"Checking MLflow model registry for: {model_name}")

            # Add force_retrain check
            force_retrain = config.get('training.force_retrain', False)

            if force_retrain:
                logger.info("Force retrain enabled - skipping model registry check")
                skip_training = False
            else:
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
        # 5. Training Loop
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

            # Get git commit information
            try:
                current_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
                commit_short = current_commit[:8]
            except:
                current_commit = "unknown"
                commit_short = "unknown"

            # Enhanced model description with metadata
            model_description = f"""Air Quality Prediction Model for {station}

            Model Details:
            - Type: {model_type}
            - Epochs Trained: {len(train_losses)}
            - Final Train Loss: {train_losses[-1]:.6f}
            - Final Val Loss: {val_losses[-1]:.6f}
            - Early Stopping: {'Yes' if early_stopping.early_stop else 'No'}
            - Git Commit: {commit_short}

            Training Configuration:
            - Learning Rate: {training_config['learning_rate']}
            - Batch Size: {training_config['batch_size']}
            - Input Sequence Length: {preprocessing_config['input_sequence_length']}
            - Output Sequence Length: {preprocessing_config['output_sequence_length']}
            """

            # Register with description
            registered_version = mlflow_handler.register_model(
                model,
                model_name,
                "Production",
                description=model_description.strip()
            )

            # Log final model version
            mlflow_handler.log_tags({
                "registered_version": str(registered_version) if registered_version else "unknown"
            })

            logger.info(f"Model registered as version: {registered_version}")

        else:
            logger.info("Skipping training - using model from registry")
            # Initialize empty lists for visualization compatibility
            train_losses, val_losses = [], []
            train_r2_scores, val_r2_scores = [], []

        # -----------------------------
        # 6. Test Evaluation
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
        # 7. Visualization
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
                figsize=(8, 6),
                title=f"Training and Validation Loss - {station}"
            )

            if viz_config.get('save_plots', False):
                plot_path = f"{plot_dir}/loss_curves_{station}.png"
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
                    figsize=(8, 6),
                    title=f"Training and Validation R² - {station}"
                )

                if viz_config.get('save_plots', False):
                    r2_plot_path = f"{plot_dir}/r2_curves_{station}.png"
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

            # fig_forecast = plot_forecast_vs_ground_truth(
            horizons = config.get('visualization.horizons')
            for horizon in horizons:
                # Unpack both matplotlib and plotly figures
                fig_mpl, fig_plotly = plot_single_horizon_forecast(
                    timestamps=test_timestamps,
                    ground_truth=targets,
                    forecasts=predictions,
                    horizon=horizon,
                    figsize=(15, 10),
                    save_path=f"{plot_dir}/forecast_h{horizon}_{station}",
                    title=f"Forecast vs Ground Truth - {station} - Horizon {horizon}"
                )

                # MLflow: Log both plots
                mlflow_handler.log_plot(fig_mpl, f"forecast_vs_ground_truth_h{horizon}_static")

                # For plotly, you may need to save and log the HTML file
                plotly_html_path = f"{plot_dir}/forecast_h{horizon}_{station}.html"
                mlflow_handler.log_artifact(plotly_html_path, "interactive_plots")

                logger.info(f"Horizon {horizon} plots created and saved")
                plt.show()
                plt.close(fig_mpl)  # Close matplotlib figure to free memory

            # if viz_config.get('save_plots', False):
            #     forecast_plot_path = f"{plot_dir}/forecast_vs_ground_truth_{station}.png"
            #     fig_forecast.savefig(forecast_plot_path, dpi=150, bbox_inches='tight')
            #     logger.info(f"Forecast vs ground truth plot saved to {forecast_plot_path}")

            # # MLflow: Log forecast plot
            # mlflow_handler.log_plot(fig_forecast, "forecast_vs_ground_truth")
            # plt.show()

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
        fig_metrics = plot_horizon_metrics(
            horizon_metrics,
            title=f"Horizon Metrics - {station}"
        )
        mlflow_handler.log_plot(fig_metrics, "horizon_metrics")

        # -----------------------------
        # 7. Store Artifacts and Metadata
        # -----------------------------

        logger.info("Storing model artifacts and metadata...")

        # Store original dataset as MLflow artifact
        logger.info("Storing original dataset as artifact...")
        original_dataset_dir = f"{plot_dir}/original_data"
        os.makedirs(original_dataset_dir, exist_ok=True)

        # Save the original loaded dataframe
        original_data_path = f"{original_dataset_dir}/original_data_{station}.parquet"
        df.to_parquet(original_data_path, index=False)
        mlflow_handler.log_artifact(original_data_path, "original_data")
        logger.info(f"Original dataset saved to {original_data_path}")

        # Save the processed dataframe (after dropping columns)
        processed_data_path = f"{original_dataset_dir}/processed_data_{station}.parquet"
        data.to_parquet(processed_data_path, index=False)
        mlflow_handler.log_artifact(processed_data_path, "original_data")
        logger.info(f"Processed dataset saved to {processed_data_path}")

        # Save dataset metadata
        dataset_metadata = {
            "station": station,
            "original_shape": df.shape,
            "processed_shape": data.shape,
            "original_columns": df.columns.tolist(),
            "processed_columns": data.columns.tolist(),
            "dropped_columns": drop_columns,
            "features": features,
            "target": target,
            "data_path": data_path,
            "date_range": {
                "start": str(data.index.min()) if hasattr(data.index, 'min') else "unknown",
                "end": str(data.index.max()) if hasattr(data.index, 'max') else "unknown"
            }
        }

        dataset_metadata_path = f"{original_dataset_dir}/dataset_metadata_{station}.json"
        with open(dataset_metadata_path, 'w') as f:
            json.dump(dataset_metadata, f, indent=2, default=str)
        mlflow_handler.log_artifact(dataset_metadata_path, "original_data")
        logger.info(f"Dataset metadata saved to {dataset_metadata_path}")

        # MLflow: Log data info
        mlflow_handler.log_metrics({
            "data_shape_rows": df.shape[0],
            "data_shape_cols": df.shape[1],
            "num_features": len(features)
        })

        # Get station name from config
        station = config.get('data.station')

        # Prepare metadata
        metadata = {
            "git_commit": current_commit,
            "git_commit_short": commit_short,
            "model_version": model_version if model_version else "new",
            "early_stopping_triggered": early_stopping.early_stop if not skip_training else False,
            "last_epoch": early_stopping.counter if not skip_training else 0,
            "total_epochs_trained": len(train_losses) if not skip_training else 0,
            "station": station,
            "model_type": model_type,
            "training_config": training_config,
            "preprocessing_config": preprocessing_config,
            "final_train_loss": train_losses[-1] if not skip_training and train_losses else None,
            "final_val_loss": val_losses[-1] if not skip_training and val_losses else None,
            "model_name": model_name
        }

        # Log metadata as MLflow tags
        mlflow_handler.log_tags({
            "git_commit": commit_short,
            "model_version": str(metadata["model_version"]),
            "early_stopping": str(metadata["early_stopping_triggered"]),
            "last_epoch": str(metadata["last_epoch"]),
            "station": station,
            "model_type": model_type
        })

        # Create artifacts directory
        artifacts_dir = f"{plot_dir}/artifacts"
        os.makedirs(artifacts_dir, exist_ok=True)

        # Save metadata as JSON artifact
        metadata_path = f"{artifacts_dir}/training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        mlflow_handler.log_artifact(metadata_path, "metadata")
        logger.info(f"Training metadata saved to {metadata_path}")

        # Save model weights as artifact
        if not skip_training:
            #model_weights_path = f"{artifacts_dir}/model_weights_{station}_{model_type}.pth"
            model_weights_path = f"{artifacts_dir}/{model_name}.pth"
            torch.save(model.state_dict(), model_weights_path)
            mlflow_handler.log_artifact(model_weights_path, "model_weights")
            logger.info(f"Model weights saved to {model_weights_path}")

        # Save scaler as artifact
        scaler_path = f"{artifacts_dir}/scaler_{station}.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(prepared_data['scaler'], f)
        mlflow_handler.log_artifact(scaler_path, "preprocessing")
        logger.info(f"Scaler saved to {scaler_path}")

        # Save processed datasets as artifacts
        datasets_dir = f"{artifacts_dir}/datasets"
        os.makedirs(datasets_dir, exist_ok=True)

        # Save train/val/test data splits
        np.save(f"{datasets_dir}/X_train.npy", prepared_data['X_train'])
        np.save(f"{datasets_dir}/y_train.npy", prepared_data['y_train'])
        np.save(f"{datasets_dir}/X_val.npy", prepared_data['X_val'])
        np.save(f"{datasets_dir}/y_val.npy", prepared_data['y_val'])
        np.save(f"{datasets_dir}/X_test.npy", prepared_data['X_test'])
        np.save(f"{datasets_dir}/y_test.npy", prepared_data['y_test'])

        # Save timestamps
        with open(f"{datasets_dir}/test_timestamps.pkl", 'wb') as f:
            pickle.dump(prepared_data['test_timestamps'], f)

        mlflow_handler.log_artifact(datasets_dir, "datasets")
        logger.info(f"Dataset splits saved to {datasets_dir}")

        # Save predictions and targets from test evaluation
        results_dir = f"{artifacts_dir}/results"
        os.makedirs(results_dir, exist_ok=True)

        if 'predictions' in locals() and 'targets' in locals():
            np.save(f"{results_dir}/test_predictions.npy", predictions)
            np.save(f"{results_dir}/test_targets.npy", targets)

        if 'horizon_metrics' in locals():
            with open(f"{results_dir}/horizon_metrics.pkl", 'wb') as f:
                pickle.dump(horizon_metrics, f)

        # Save training history if available
        if not skip_training and 'train_losses' in locals():
            training_history = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_r2_scores': train_r2_scores if 'train_r2_scores' in locals() else [],
                'val_r2_scores': val_r2_scores if 'val_r2_scores' in locals() else []
            }
            with open(f"{results_dir}/training_history.pkl", 'wb') as f:
                pickle.dump(training_history, f)

        mlflow_handler.log_artifact(results_dir, "results")
        logger.info(f"Results saved to {results_dir}")

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