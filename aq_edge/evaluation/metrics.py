import numpy as np
from aq_edge.utils.logging import LoggerHandler

# Initialize logger at module level
logger = LoggerHandler(__name__)


def calculate_horizon_metrics(predictions, truths):
    """Calculate metrics for each forecast horizon"""
    logger.info(f"Input shapes - predictions: {predictions.shape}, truths: {truths.shape}")

    if predictions.shape != truths.shape:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs truths {truths.shape}")

    output_len = predictions.shape[1]
    logger.info(f"Calculating metrics for {output_len} horizons")

    metrics = {
        'RMSE': [],
        'MAE': [],
        'R2': [],
        'MAPE': [],
    }

    for h in range(output_len):
        logger.info(f"Processing horizon {h}")
        y_true = truths[:, h]
        y_pred = predictions[:, h]

        logger.info(f"  Horizon {h}: y_true shape {y_true.shape}, y_pred shape {y_pred.shape}")

        # RMSE
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        # MAE
        mae = np.mean(np.abs(y_true - y_pred))

        # R² (Coefficient of Determination)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        # MAPE (handle division by zero)
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100


        metrics['RMSE'].append(rmse)
        metrics['MAE'].append(mae)
        metrics['R2'].append(r2)
        metrics['MAPE'].append(mape)

        logger.info(f"  Horizon {h} metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

    return metrics
