import argparse
import json
import os
from typing import Any, Dict, Iterator, Optional

import yaml


class ConfigHandler:
    """Load YAML config, then apply optional Optuna and CLI overrides."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        argv: Optional[list[str]] = None,
    ):
        if config_path is None:
            config_path = self._find_default_config()

        self.config_path = config_path
        self.config = self._load_config()
        self._apply_cli_overrides(argv)

    def _find_default_config(self) -> str:
        """Find the default project configuration file."""
        possible_paths = [
            "config.yaml",
            "config/config.yaml",
            "configs/config.yaml",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            "No default config file found. Please specify config_path."
        )

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}"
            )
        except yaml.YAMLError as error:
            raise ValueError(
                f"Error parsing config file: {error}"
            ) from error

        if not isinstance(config, dict):
            raise ValueError(
                "The YAML configuration must contain a mapping."
            )

        return config

    @staticmethod
    def _leaf_keys(
        section: Dict[str, Any],
        prefix: str = "",
    ) -> Iterator[str]:
        """Yield dotted paths for all non-mapping configuration values."""
        for key, value in section.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                yield from ConfigHandler._leaf_keys(value, path)
            else:
                yield path

    def _apply_cli_overrides(self, argv: Optional[list[str]]) -> None:
        parser = argparse.ArgumentParser(
            description="Run with YAML, optional Optuna, and CLI configuration."
        )

        # Add an option for every leaf setting in the loaded config.
        for key in self._leaf_keys(self.config):
            options = [f"--{key}"]
            if key == "data.station":
                options.append("--station")

            parser.add_argument(
                *options,
                dest=key,
                metavar="VALUE",
                default=argparse.SUPPRESS,
            )

        parser.add_argument(
            "--best-params",
            dest="best_params_path",
            metavar="JSON",
            default=None,
            help="Optional path to the Optuna JSON results file.",
        )

        parsed_args = vars(parser.parse_args(argv))
        best_params_path = parsed_args.pop("best_params_path", None)

        # Use the CLI station, if provided, when validating the Optuna file.
        cli_station = parsed_args.get("data.station")
        expected_station = (
            yaml.safe_load(cli_station)
            if cli_station is not None
            else self.get("data.station")
        )

        # Apply Optuna parameters before CLI overrides.
        if best_params_path is not None:
            self._apply_best_params(
                best_params_path,
                expected_station,
            )

        # CLI values take precedence over Optuna values.
        for key, raw_value in parsed_args.items():
            self.update(key, yaml.safe_load(raw_value))

    def _apply_best_params(
        self,
        best_params_path: str,
        expected_station: Optional[str] = None,
    ) -> None:
        """Load Optuna best parameters and apply them to the config."""
        try:
            with open(best_params_path, "r", encoding="utf-8") as file:
                best_result = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Best-parameters file not found: {best_params_path}"
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in best-parameters file "
                f"{best_params_path}: {error}"
            ) from error

        if not isinstance(best_result, dict):
            raise ValueError(
                f"Expected a JSON object in {best_params_path}"
            )

        best_params = best_result.get("best_parameters")
        if not isinstance(best_params, dict):
            raise ValueError(
                f"Expected a 'best_parameters' object in {best_params_path}"
            )

        result_station = best_result.get("station")
        station = (
            expected_station
            if expected_station is not None
            else self.get("data.station")
        )

        if result_station is not None and result_station != station:
            raise ValueError(
                f"Optuna parameters are for station {result_station!r}, "
                f"but the config specifies {station!r}"
            )

        config_keys = {
            "model_type": "model.type",
            "hidden_size": "model.hidden_size",
            "num_layers": "model.num_layers",
            "dropout": "model.dropout",
            "batch_size": "training.batch_size",
            "learning_rate": "training.learning_rate",
            "weight_decay": "training.weight_decay",
            "use_reduce_on_plateau": "training.use_reduce_on_plateau",
            "scheduler_factor": "training.scheduler_factor",
            "scheduler_patience": "training.scheduler_patience",
            "scheduler_min_lr": "training.scheduler_min_lr",
            "grad_clip_max_norm": "training.grad_clip_max_norm",
            "early_stopping_patience": "training.patience",
            "num_epochs": "training.num_epochs",
            "input_sequence_length": (
                "preprocessing.input_sequence_length"
            ),
            "output_sequence_length": (
                "preprocessing.output_sequence_length"
            ),
            "scaler": "preprocessing.scaler_type",
        }

        for parameter, config_key in config_keys.items():
            if parameter in best_params:
                self.update(config_key, best_params[parameter])

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using a dotted key."""
        value = self.config
        try:
            for part in key.split("."):
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get an entire configuration section."""
        return self.config.get(section, {})

    def update(self, key: str, value: Any) -> None:
        """Update a configuration value using a dotted key."""
        parts = key.split(".")
        config_ref = self.config

        for part in parts[:-1]:
            if part not in config_ref:
                config_ref[part] = {}
            config_ref = config_ref[part]

        config_ref[parts[-1]] = value

    def save(self, output_path: Optional[str] = None) -> None:
        """Save the current configuration as YAML."""
        path = output_path or self.config_path
        with open(path, "w", encoding="utf-8") as file:
            yaml.safe_dump(
                self.config,
                file,
                default_flow_style=False,
            )