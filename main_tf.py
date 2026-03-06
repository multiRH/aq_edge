import os
from dotenv import load_dotenv
load_dotenv(override=True)

import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import tempfile

from aq_edge.utils.logging import LoggerHandler
from aq_edge.utils.config import ConfigHandler
from aq_edge.utils.mlflow_handler import MLflowHandler
from aq_edge.utils.visualization import plot_single_horizon_forecast
from aq_edge.evaluation.metrics import calculate_horizon_metrics


class TFLiteConverter:
    """Convert PyTorch models to TensorFlow Lite for edge deployment"""

    def __init__(self, logger):
        self.logger = logger

    def pytorch_to_tflite(self,
                         pytorch_model: nn.Module,
                         input_shape: Tuple,
                         calibration_data: np.ndarray,
                         quantize: bool = True) -> bytes:
        """
        Convert PyTorch model to TFLite format with optional quantization

        Args:
            pytorch_model: PyTorch model to convert
            input_shape: Shape of input tensor (batch_size, seq_len, features)
            calibration_data: Data for post-training quantization
            quantize: Whether to apply int8 quantization

        Returns:
            TFLite model as bytes
        """
        self.logger.info("Converting PyTorch model to TensorFlow Lite...")

        # Step 1: Convert PyTorch to ONNX
        self.logger.info("Step 1: Converting PyTorch to ONNX...")
        pytorch_model.eval()
        pytorch_model.cpu()

        dummy_input = torch.randn(input_shape)
        onnx_path = tempfile.NamedTemporaryFile(delete=False, suffix='.onnx').name

        torch.onnx.export(
            pytorch_model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            opset_version=11
        )
        self.logger.info(f"ONNX model saved to {onnx_path}")

        # Step 2: Convert ONNX to TensorFlow
        self.logger.info("Step 2: Converting ONNX to TensorFlow...")
        try:
            import onnx
            from onnx_tf.backend import prepare

            onnx_model = onnx.load(onnx_path)
            tf_rep = prepare(onnx_model)

            tf_model_dir = tempfile.mkdtemp()
            tf_rep.export_graph(tf_model_dir)
            self.logger.info(f"TensorFlow model saved to {tf_model_dir}")

        except ImportError:
            self.logger.error("onnx-tf not installed. Install with: pip install onnx-tf")
            raise

        # Step 3: Convert TensorFlow to TFLite
        self.logger.info("Step 3: Converting TensorFlow to TFLite...")
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)

        if quantize:
            self.logger.info("Applying post-training quantization...")

            # Enable quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            # Representative dataset for quantization
            def representative_dataset():
                for i in range(min(100, len(calibration_data))):
                    yield [calibration_data[i:i+1].astype(np.float32)]

            converter.representative_dataset = representative_dataset

            # Full integer quantization for microcontrollers
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

            self.logger.info("Quantization configuration set")

        tflite_model = converter.convert()
        self.logger.info("TFLite conversion completed")

        # Cleanup
        os.remove(onnx_path)

        return tflite_model

    def get_tflite_model_size(self, tflite_model: bytes) -> float:
        """Calculate TFLite model size in MB"""
        return len(tflite_model) / (1024 * 1024)


class TFLiteInference:
    """Run inference using TFLite models"""

    def __init__(self, tflite_model: bytes, logger):
        self.logger = logger
        self.interpreter = tf.lite.Interpreter(model_content=tflite_model)
        self.interpreter.allocate_tensors()

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.logger.info("TFLite interpreter initialized")
        self.logger.info(f"Input shape: {self.input_details[0]['shape']}")
        self.logger.info(f"Output shape: {self.output_details[0]['shape']}")

    def run_inference(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on input data

        Args:
            input_data: Input numpy array

        Returns:
            Predictions as numpy array
        """
        predictions = []

        for i in range(len(input_data)):
            # Prepare input
            input_tensor = input_data[i:i+1].astype(np.float32)

            # Handle quantized input
            if self.input_details[0]['dtype'] == np.int8:
                input_scale, input_zero_point = self.input_details[0]['quantization']
                input_tensor = (input_tensor / input_scale + input_zero_point).astype(np.int8)

            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.interpreter.invoke()

            # Get output
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

            # Handle quantized output
            if self.output_details[0]['dtype'] == np.int8:
                output_scale, output_zero_point = self.output_details[0]['quantization']
                output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

            predictions.append(output_data[0])

        return np.array(predictions)


class ModelInference:
    """Handle model inference and comparison"""

    def __init__(self, config: ConfigHandler, logger: LoggerHandler):
        self.config = config
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() and
                                   config.get('device.use_cuda', True) else "cpu")

    def load_model_from_registry(self, model_name: str,
                                version: Optional[str] = None) -> nn.Module:
        """Load model from MLflow registry"""
        mlflow_handler = MLflowHandler(
            experiment_name=f"air_quality_{self.config.get('data.station', 'default')}",
            enabled=True
        )

        if version is None:
            version = mlflow_handler.get_latest_model_version(model_name)

        model_uri = f"models:/{model_name}/{version}"
        self.logger.info(f"Loading model from: {model_uri}")

        model = mlflow_handler.load_model(model_uri)
        if model is None:
            raise ValueError(f"Failed to load model {model_name} version {version}")

        self.logger.info(f"Model loaded successfully: {model_name} v{version}")
        return model

    def load_test_data(self, artifacts_path: str) -> Tuple[np.ndarray, np.ndarray, list]:
        """Load test sequence data"""
        self.logger.info(f"Loading test data from: {artifacts_path}")

        X_test = np.load(f"{artifacts_path}/datasets/X_test.npy")
        y_test = np.load(f"{artifacts_path}/datasets/y_test.npy")

        with open(f"{artifacts_path}/datasets/test_timestamps.pkl", 'rb') as f:
            timestamps = pickle.load(f)

        self.logger.info(f"Test data loaded - X_test: {X_test.shape}, y_test: {y_test.shape}")
        return X_test, y_test, timestamps

    def run_inference(self, model: nn.Module,
                     X_test: np.ndarray,
                     device: Optional[torch.device] = None) -> np.ndarray:
        """Run inference on test data"""
        if device is None:
            device = self.device

        model.eval()
        model.to(device)

        X_tensor = torch.FloatTensor(X_test).to(device)

        self.logger.info("Running inference...")
        with torch.no_grad():
            predictions = model(X_tensor)

        predictions_np = predictions.cpu().numpy()
        self.logger.info(f"Inference completed - predictions shape: {predictions_np.shape}")

        return predictions_np

    def compare_predictions(self, pred_original: np.ndarray,
                          pred_quantized: np.ndarray,
                          targets: np.ndarray) -> dict:
        """Compare original vs quantized model predictions"""
        metrics_original = calculate_horizon_metrics(pred_original, targets)
        metrics_quantized = calculate_horizon_metrics(pred_quantized, targets)

        pred_diff = np.abs(pred_original - pred_quantized)
        mean_diff = np.mean(pred_diff)
        max_diff = np.max(pred_diff)

        comparison = {
            'original_metrics': metrics_original,
            'quantized_metrics': metrics_quantized,
            'mean_prediction_diff': mean_diff,
            'max_prediction_diff': max_diff,
            'diff_per_horizon': np.mean(pred_diff, axis=0).tolist()
        }

        self.logger.info(f"Prediction comparison - Mean diff: {mean_diff:.6f}, Max diff: {max_diff:.6f}")

        return comparison


def main():
    logger = LoggerHandler(__name__)
    logger.info("Starting TFLite model inference and quantization pipeline")

    try:
        config = ConfigHandler()
        station = config.get('data.station')

        output_dir = Path("inference_results")
        output_dir.mkdir(exist_ok=True)

        inference = ModelInference(config, logger)

        # Get model name
        model_type = config.get('model.type', 'AttentionLSTM')
        num_epochs = config.get('training.num_epochs')
        learning_rate = config.get('training.learning_rate')
        batch_size = config.get('training.batch_size')

        model_name = f"{station}_{model_type}_e{num_epochs}_lr{str(learning_rate).replace('.', '')}_b{batch_size}_model"

        # Load model from registry
        logger.info(f"Loading model: {model_name}")
        original_model = inference.load_model_from_registry(model_name)

        # Load test data
        artifacts_path = config.get('visualization.plot_dir', 'plots') + "/artifacts"
        X_test, y_test, timestamps = inference.load_test_data(artifacts_path)

        # Run inference with original PyTorch model
        logger.info("Running inference with original PyTorch model...")
        pred_original = inference.run_inference(original_model, X_test)

        # Get PyTorch model size
        pytorch_model_path = output_dir / "temp_pytorch_model.pth"
        torch.save(original_model.state_dict(), pytorch_model_path)
        size_pytorch = os.path.getsize(pytorch_model_path) / (1024 * 1024)
        os.remove(pytorch_model_path)

        # Convert to TFLite
        logger.info("Converting PyTorch model to TFLite...")
        converter = TFLiteConverter(logger)

        input_shape = (1, X_test.shape[1], X_test.shape[2])
        calibration_data = X_test[:100]

        tflite_model = converter.pytorch_to_tflite(
            original_model,
            input_shape,
            calibration_data,
            quantize=True
        )

        # Save TFLite model
        tflite_model_path = output_dir / f"{model_name}_quantized.tflite"
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        logger.info(f"TFLite model saved to {tflite_model_path}")

        # Get TFLite model size
        size_tflite = converter.get_tflite_model_size(tflite_model)
        compression_ratio = size_pytorch / size_tflite

        logger.info(f"PyTorch model size: {size_pytorch:.2f} MB")
        logger.info(f"TFLite model size: {size_tflite:.2f} MB")
        logger.info(f"Compression ratio: {compression_ratio:.2f}x")

        # Run inference with TFLite model
        logger.info("Running inference with TFLite model...")
        tflite_inference = TFLiteInference(tflite_model, logger)
        pred_tflite = tflite_inference.run_inference(X_test)

        # Compare predictions
        comparison = inference.compare_predictions(pred_original, pred_tflite, y_test)

        # Save comparison results
        comparison_results = {
            'model_name': model_name,
            'pytorch_size_mb': size_pytorch,
            'tflite_size_mb': size_tflite,
            'compression_ratio': compression_ratio,
            'comparison_metrics': comparison
        }

        results_path = output_dir / f"tflite_quantization_results_{station}.json"
        with open(results_path, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        logger.info(f"Comparison results saved to {results_path}")

        # Create visualizations
        horizons = config.get('visualization.horizons', [1, 3, 6, 12, 24])

        for horizon in horizons:
            logger.info(f"Creating plots for horizon {horizon}...")

            # Plot PyTorch model predictions
            fig_pytorch_mpl, _ = plot_single_horizon_forecast(
                timestamps=timestamps,
                ground_truth=y_test,
                forecasts=pred_original,
                horizon=horizon,
                figsize=(15, 10),
                save_path=str(output_dir / f"pytorch_h{horizon}_{station}"),
                title=f"PyTorch Model - {station} - Horizon {horizon}"
            )

            # Plot TFLite model predictions
            fig_tflite_mpl, _ = plot_single_horizon_forecast(
                timestamps=timestamps,
                ground_truth=y_test,
                forecasts=pred_tflite,
                horizon=horizon,
                figsize=(15, 10),
                save_path=str(output_dir / f"tflite_h{horizon}_{station}"),
                title=f"TFLite Model - {station} - Horizon {horizon}"
            )

            # Create comparison plot
            fig_compare = plt.figure(figsize=(15, 10))

            horizon_idx = horizon - 1
            gt_values = y_test[:, horizon_idx]
            pred_pytorch_values = pred_original[:, horizon_idx]
            pred_tflite_values = pred_tflite[:, horizon_idx]

            shifted_timestamps = timestamps[horizon:]
            aligned_gt = gt_values[:len(shifted_timestamps)]
            aligned_pytorch = pred_pytorch_values[:len(shifted_timestamps)]
            aligned_tflite = pred_tflite_values[:len(shifted_timestamps)]

            plt.plot(shifted_timestamps, aligned_gt, label='Ground Truth',
                    color='blue', linewidth=2, alpha=0.8)
            plt.plot(shifted_timestamps, aligned_pytorch, label='PyTorch Model',
                    color='green', linewidth=1.5, alpha=0.7, linestyle='--')
            plt.plot(shifted_timestamps, aligned_tflite, label='TFLite Model (Quantized)',
                    color='red', linewidth=1.5, alpha=0.7, linestyle=':')

            plt.xlabel('Time')
            plt.ylabel('CO2 Value')
            plt.title(f'PyTorch vs TFLite Comparison - {station} - Horizon {horizon}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()

            compare_path = output_dir / f"comparison_h{horizon}_{station}.png"
            plt.savefig(compare_path, dpi=150, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {compare_path}")

            plt.close('all')

        # Print summary
        logger.info("="*70)
        logger.info("TFLITE QUANTIZATION SUMMARY")
        logger.info("="*70)
        logger.info(f"Model: {model_name}")
        logger.info(f"PyTorch size: {size_pytorch:.2f} MB")
        logger.info(f"TFLite size: {size_tflite:.2f} MB")
        logger.info(f"Compression: {compression_ratio:.2f}x reduction")
        logger.info(f"Mean prediction difference: {comparison['mean_prediction_diff']:.6f}")
        logger.info(f"Max prediction difference: {comparison['max_prediction_diff']:.6f}")
        logger.info("="*70)

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()