import numpy as np
import pandas as pd
from matplotlib import pyplot as plt




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


def plot_horizon_metrics(metrics):
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