import argparse
import os
from typing import Any, Dict, Iterator, Optional

import yaml


class ConfigHandler:
    """Load YAML configuration and apply command-line overrides."""

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
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        except yaml.YAMLError as error:
            raise ValueError(f"Error parsing config file: {error}") from error

        if not isinstance(config, dict):
            raise ValueError("The YAML configuration must contain a mapping.")

        return config

    @staticmethod
    def _leaf_keys(section: Dict[str, Any], prefix: str = "") -> Iterator[str]:
        """Yield dotted paths for all non-mapping configuration values."""
        for key, value in section.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                yield from ConfigHandler._leaf_keys(value, path)
            else:
                yield path

    def _apply_cli_overrides(self, argv: Optional[list[str]]) -> None:
        parser = argparse.ArgumentParser(
            description="Run the application with YAML configuration overrides."
        )

        for key in self._leaf_keys(self.config):
            parser.add_argument(
                f"--{key}",
                dest=key,
                metavar="VALUE",
                default=argparse.SUPPRESS,
            )

        for key, raw_value in vars(parser.parse_args(argv)).items():
            self.update(key, yaml.safe_load(raw_value))

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
            yaml.safe_dump(self.config, file, default_flow_style=False)