# Updated main.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import traceback
import os

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# Import all modules
from aq_edge.utils.logging import LoggerHandler
from aq_edge.utils.config import ConfigHandler
from aq_edge.datautils.air_quality_analysis import load_station_data
from aq_edge.datautils.preprocessing import prepare_data
from aq_edge.modelzoo.lstm import BaseLSTM, AttentionLSTM
from aq_edge.modelzoo.model_factory import EarlyStopping
from aq_edge.evaluation.metrics import calculate_horizon_metrics
from aq_edge.utils.visualization import plot_horizon_predictions, plot_horizon_metrics

def main():
    # Load configuration
    config = ConfigHandler()

    # Set random seeds for reproducibility
    seed = config.get('training.random_seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize logger
    logger_name = config.get('logging.name', 'main')
    logger = LoggerHandler(logger_name)

    logger.info("="*70)
    logger.info("AIR QUALITY MODEL TRAINING AND EVALUATION")
    logger.info("="*70)

    try:
        # -----------------------------
        # 1. Load data
        # -----------------------------
        station = config.get('data.station')
        data_path = config.get('data.data_path').format(station=station)
        drop_columns = config.get('data.drop_columns', [])

        logger.info(f"Loading air quality data for station: {station}")
        df = pd.read_parquet(data_path)

        print("Original data shape:", df.shape)
        print("Columns:", df.columns.tolist())

        # Drop specified columns
        data = df.drop(columns=drop_columns)

        # Define features and target for modeling
        features = config.get('data.features')
        target = config.get('data.target')

        logger.info(f"Preparing data with features of station {station}: {features} and target: {target}")

        # -----------------------------
        # 2. Prepare data
        # -----------------------------
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

        # -----------------------------
        # 3. Initialize model
        # -----------------------------
        use_cuda = config.get('device.use_cuda', True)
        device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        print(f"Using device: {device}")

        model_type = config.get('model.type', 'AttentionLSTM')
        input_size = len(features)
        output_size = preprocessing_config['output_sequence_length']

        if model_type == 'AttentionLSTM':
            model = AttentionLSTM(input_size=input_size, output_size=output_size).to(device)
        else:
            model = BaseLSTM(input_size=input_size, output_size=output_size).to(device)

        # Initialize optimizer and criterion
        training_config = config.get_section('training')
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
        )
        criterion = nn.MSELoss()

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # -----------------------------
        # 4. Training Loop
        # -----------------------------
        num_epochs = training_config['num_epochs']
        patience = training_config['patience']
        early_stopping = EarlyStopping(patience=patience)
        train_losses, val_losses = [], []

        print("Starting training...")
        for epoch in range(num_epochs):
            model.train()
            batch_losses = []

            for xb, yb in train_dataloader:
                xb, yb = xb.to(device), yb.to(device)

                if epoch == 0 and len(batch_losses) == 0:
                    print(f"Input shape: {xb.shape}")
                    print(f"Target shape: {yb.shape}")

                optimizer.zero_grad()
                out = model(xb)

                if yb.dim() > 2:
                    yb = yb.squeeze(-1)

                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

            train_loss = np.mean(batch_losses)
            train_losses.append(train_loss)

            # Validation
            model.eval()
            val_batch_losses = []
            with torch.no_grad():
                for xb, yb in validation_dataloader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)

                    if yb.dim() > 2:
                        yb = yb.squeeze(-1)

                    loss = criterion(out, yb)
                    val_batch_losses.append(loss.item())

            val_loss = np.mean(val_batch_losses)
            val_losses.append(val_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Check early stopping
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break

        # -----------------------------
        # 5. Test Evaluation
        # -----------------------------
        print("Evaluating on test set...")
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
        print(f"Test Loss: {test_loss:.6f}")

        # Convert to numpy arrays for analysis
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        print(f"Test predictions shape: {predictions.shape}")
        print(f"Test targets shape: {targets.shape}")

        # -----------------------------
        # 6. Visualization
        # -----------------------------
        viz_config = config.get_section('visualization')

        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.grid(True)

        if viz_config.get('save_plots', False):
            plot_dir = viz_config.get('plot_dir', 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(f"{plot_dir}/training_loss.png")

        plt.show()

        print("Calculating metrics per horizon...")
        horizon_metrics = calculate_horizon_metrics(predictions, targets)

        # Print summary statistics
        print("\nHorizon Metrics Summary:")
        for metric, values in horizon_metrics.items():
            print(f"{metric}: Mean={np.mean(values):.4f}, Std={np.std(values):.4f}")

        # Plot metrics per horizon
        print("Plotting metrics per horizon...")
        plot_horizon_metrics(horizon_metrics)

    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()