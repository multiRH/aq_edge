import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigHandler:
    """Handler for loading and managing configuration files."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config handler.

        Args:
            config_path (str, optional): Path to config file. If None, looks for default config.
        """
        if config_path is None:
            config_path = self._find_default_config()

        self.config_path = config_path
        self.config = self._load_config()

    def _find_default_config(self) -> str:
        """Find default config file in project directory."""
        possible_paths = [
            'config.yaml',
            'config/config.yaml',
            'configs/config.yaml'
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        raise FileNotFoundError("No default config file found. Please specify config_path.")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing config file: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with dots).

        Args:
            key (str): Configuration key (e.g., 'model.input_size' or 'data.features')
            default (Any): Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self.config.get(section, {})

    def update(self, key: str, value: Any) -> None:
        """Update configuration value."""
        keys = key.split('.')
        config_ref = self.config

        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]

        config_ref[keys[-1]] = value

    def save(self, output_path: Optional[str] = None) -> None:
        """Save configuration to file."""
        path = output_path or self.config_path
        with open(path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)