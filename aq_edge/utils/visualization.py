import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import List, Optional, Tuple


# 2.8 Plot Horizon-wise Predictions vs. Truth
def plot_horizon_predictions(predictions, truths, horizon_idx=0):
    """Plot predictions vs. ground truth for a specific horizon"""
    y_true = truths[:, horizon_idx]
    y_pred = predictions[:, horizon_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Scatter plot
    ax1.scatter(y_true, y_pred, alpha=0.6, s=20)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('Ground Truth')
    ax1.set_ylabel('Predictions')
    ax1.set_title(f'Predictions vs Ground Truth (Horizon {horizon_idx + 1})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Time series comparison (first 100 samples)
    n_display = min(100, len(y_true))
    x_axis = range(n_display)
    ax2.plot(x_axis, y_true[:n_display], 'o-', label='Ground Truth', color='green', alpha=0.8)
    ax2.plot(x_axis, y_pred[:n_display], 's-', label='Predictions', color='red', alpha=0.8)
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Yield Value')
    ax2.set_title(f'Time Series Comparison (Horizon {horizon_idx + 1}, First {n_display} samples)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_horizon_metrics(metrics, title=None):
    """Plot metrics across forecast horizons"""
    # Check if metrics are empty
    if not metrics or all(len(values) == 0 for values in metrics.values()):
        print("No metrics to plot - all metric lists are empty")
        return

    # Get the actual length from the first non-empty metric
    output_len = None
    for metric_name, values in metrics.items():
        if len(values) > 0:
            output_len = len(values)
            break

    if output_len is None:
        print("No valid metrics found")
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.flatten()

    # Set main title if provided
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')

    for i, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[i]

        # Skip empty metrics
        if len(values) == 0:
            ax.text(0.5, 0.5, f'No {metric_name} data',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
            ax.set_title(f'{metric_name} per Forecast Horizon')
            continue

        horizons = range(1, len(values) + 1)

        ax.plot(horizons, values, 'o-', linewidth=2, markersize=8, color='steelblue')
        ax.set_xlabel('Forecast Horizon')
        ax.set_ylabel(f'{metric_name} Value')
        ax.set_title(f'{metric_name} per Forecast Horizon')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(horizons)

        # Add value annotations
        for j, v in enumerate(values):
            ax.annotate(f'{v:.3f}', (j + 1, v), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()
    return fig


def plot_loss_curves(train_losses: List[float],
                    val_losses: List[float],
                    figsize: Tuple[int, int] = (8, 6),
                    save_path: Optional[str] = None,
                    title: Optional[str] = None) -> plt.Figure:
    """
    Plot training and validation MSE loss curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        figsize: Figure size as (width, height)
        save_path: Optional path to save the plot
        title: Optional custom title for the plot

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(train_losses, label="Train Loss", color='blue', linewidth=2)
    ax.plot(val_losses, label="Val Loss", color='red', linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")

    # Use custom title if provided, otherwise use default
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Training & Validation Loss")

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig

def plot_r2_curves(train_r2_scores: List[float],
                   val_r2_scores: List[float],
                   figsize: Tuple[int, int] = (8, 6),
                   save_path: Optional[str] = None,
                   title: Optional[str] = None) -> plt.Figure:
    """
    Plot training and validation R² score curves.

    Args:
        train_r2_scores: List of training R² scores per epoch
        val_r2_scores: List of validation R² scores per epoch
        figsize: Figure size as (width, height)
        save_path: Optional path to save the plot
        title: Optional custom title for the plot

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(train_r2_scores, label="Train R²", color='green', linewidth=2)
    ax.plot(val_r2_scores, label="Val R²", color='orange', linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R² Score")

    # Use custom title if provided, otherwise use default
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Training & Validation R² Score")

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_forecast_vs_ground_truth(timestamps: List,
                                 ground_truth: np.ndarray,
                                 forecasts: np.ndarray,
                                 horizons: List[int] = [1, 3, 9, 12],
                                 figsize: Tuple[int, int] = (15, 10),
                                 save_path: Optional[str] = None,
                                 title: Optional[str] = None) -> plt.Figure:
    """
    Plot forecasted values vs ground truth for specific horizons with timestamps.
    Each horizon is shifted by its respective time steps.

    Args:
        timestamps: List of timestamps corresponding to the base predictions
        ground_truth: Ground truth values of shape (n_samples, n_horizons)
        forecasts: Forecasted values of shape (n_samples, n_horizons)
        horizons: List of horizon indices to plot (1-indexed)
        figsize: Figure size as (width, height)
        save_path: Optional path to save the plot
        title: Optional custom title for the plot

    Returns:
        matplotlib Figure object
    """
    n_horizons = len(horizons)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # Set main title if provided
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')

    # Align timestamps with the actual number of samples
    n_samples = ground_truth.shape[0]
    base_timestamps = timestamps[:n_samples] if len(timestamps) > n_samples else timestamps

    for i, horizon in enumerate(horizons):
        ax = axes[i]
        horizon_idx = horizon - 1  # Convert to 0-indexed

        # Extract data for this horizon
        gt_values = ground_truth[:, horizon_idx]
        pred_values = forecasts[:, horizon_idx]

        # Create shifted timestamps for this horizon
        # Each horizon represents a forecast `horizon` steps into the future
        shifted_timestamps = base_timestamps[horizon:]  # Shift by horizon steps

        # Align the data with the shifted timestamps
        aligned_gt = gt_values[:len(shifted_timestamps)]
        aligned_pred = pred_values[:len(shifted_timestamps)]

        # Create time series plot
        ax.plot(shifted_timestamps, aligned_gt, label='Ground Truth', color='blue', linewidth=1.5, alpha=0.8)
        ax.plot(shifted_timestamps, aligned_pred, label='Forecast', color='red', linewidth=1.5, alpha=0.8)

        ax.set_xlabel('Time')
        ax.set_ylabel('CO2 Value')
        ax.set_title(f'Forecast vs Ground Truth - Horizon {horizon} (Shift: +{horizon})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig