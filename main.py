import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import traceback

from click.core import batch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# Import all modules
from aq_edge.utils.logging import LoggerHandler
from aq_edge.datautils.air_quality_analysis import load_station_data
from aq_edge.datautils.preprocessing import prepare_data
from aq_edge.modelzoo.lstm import BaseLSTM, AttentionLSTM
from aq_edge.modelzoo.model_factory import EarlyStopping
from aq_edge.evaluation.metrics import calculate_horizon_metrics
from aq_edge.utils.visualization import plot_horizon_predictions, plot_horizon_metrics

def main():
    pass

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize logger
    logger = LoggerHandler('main')
    #mlflow

    logger.info("="*70)
    logger.info("AIR QUALITY MODEL TRAINING AND EVALUATION")
    logger.info("="*70)

    try:
        # -----------------------------
        # 1. Load data
        # -----------------------------
        station = 'USAM'
        logger.info("Loading air quality data...")
        # Load data for a specific station (e.g., 'USAM')
        df = pd.read_parquet(f'data/air/processed/{station}.parquet')

        print("Original data shape:", df.shape)
        print("Columns:", df.columns.tolist())

        data = df.drop(columns=['ICA'])

        # Define features and target for modeling
        features = ["TEM", "HUM", "VOC"]
        target = "CO2"

        logger.info(f"Preparing data with features: {features} and target: {target}")

        # -----------------------------
        # 2. Prepare data
        # -----------------------------

        input_sequence_length = 12  # 24 hours of history
        output_sequence_length = 6  # Predict next hour
        batch_size     = 32
        train_ratio     = 0.7
        validation_ratio= 0.15
        batch_size     = 32

        # Use the new prepare_data function
        prepared_data = prepare_data(
            data=data,
            features=features,
            target=target,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            input_sequence_length=input_sequence_length,  # 24 hours of history
            output_sequence_length=output_sequence_length,  # Predict next hour
            batch_size=batch_size
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model = AttentionLSTM(input_size=len(features), output_size=output_sequence_length).to(device)
        # model = BaseLSTM(input_size=len(features), output_size=output_sequence_length).to(device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = nn.MSELoss()

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # -----------------------------
        # 4. Training Loop
        # -----------------------------
        num_epochs = 25
        early_stopping = EarlyStopping(patience=5)
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
            # check early stopping
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

                # Store predictions and targets for further analysis
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
        # 10. Plot Training/Validation Loss
        # -----------------------------
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

        print("Calculating metrics per horizon...")
        horizon_metrics = calculate_horizon_metrics(predictions, targets)

        # Print summary statistics
        print("\nHorizon Metrics Summary:")
        for metric, values in horizon_metrics.items():
            print(f"{metric}: Mean={np.mean(values):.4f}, Std={np.std(values):.4f}")

        # -----------------------------
        # 14. Plot Metrics per Horizon
        # -----------------------------
        print("Plotting metrics per horizon...")
        plot_horizon_metrics(horizon_metrics)


    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        traceback.print_exc()

