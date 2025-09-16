import logging
import os
from datetime import datetime
from typing import Dict, Optional
import atexit
import glob

class LoggerHandler:
    # Class variables to track all log files
    _log_files: Dict[str, str] = {}
    _handlers: Dict[str, logging.FileHandler] = {}
    _cleanup_registered = False

    def __init__(self, name=__name__, level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.module_name = name

        # Only add handlers if they don't already exist
        if not self.logger.handlers:
            # Create artifacts directory if it doesn't exist
            artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'artifacts')
            os.makedirs(artifacts_dir, exist_ok=True)

            # Create log filename with timestamp and module name
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            module_name = name.split('.')[-1] if '.' in name else name
            # Replace __main__ with main for cleaner filename
            if module_name == '__main__':
                module_name = 'main'

            log_filename = f"{module_name}_{timestamp}.log"
            log_path = os.path.join(artifacts_dir, log_filename)

            # Store log file path in class variable
            LoggerHandler._log_files[name] = log_path

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s',
                                                datefmt='%Y-%m-%d %H:%M:%S')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

            # File handler
            file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
            file_handler.setLevel(level)
            file_formatter = logging.Formatter('[%(asctime)s] %(name)s - %(levelname)s: %(message)s',
                                             datefmt='%Y-%m-%d %H:%M:%S')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

            # Store handler reference for cleanup
            LoggerHandler._handlers[name] = file_handler

            # Register cleanup function once
            if not LoggerHandler._cleanup_registered:
                atexit.register(LoggerHandler._cleanup_logging)
                LoggerHandler._cleanup_registered = True

            # Log the file location
            self.logger.info(f"Logging to file: {log_path}")

    @classmethod
    def _cleanup_logging(cls):
        """Ensure all logs are flushed before program exit."""
        for handler in cls._handlers.values():
            try:
                handler.flush()
                handler.close()
            except Exception:
                pass

    @classmethod
    def get_log_files(cls) -> Dict[str, str]:
        """Get all log file paths for MLflow artifact logging."""
        return cls._log_files.copy()

    @classmethod
    def cleanup_unwanted_logs(cls, keep_modules: list = None):
        """Remove log files from modules that shouldn't have them."""
        if keep_modules is None:
            keep_modules = ['__main__']  # Only keep main.py logs by default

        artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'artifacts')

        # Find all log files in artifacts directory
        log_pattern = os.path.join(artifacts_dir, "*.log")
        all_log_files = glob.glob(log_pattern)

        for log_file in all_log_files:
            filename = os.path.basename(log_file)
            # Extract module name from filename (before the timestamp)
            module_part = filename.split('_')[0]

            # Remove unwanted log files
            if module_part not in keep_modules and module_part != 'main':
                try:
                    os.remove(log_file)
                    print(f"Removed unwanted log file: {filename}")

                    # Also remove from tracking dictionaries
                    keys_to_remove = []
                    for key, path in cls._log_files.items():
                        if path == log_file:
                            keys_to_remove.append(key)

                    for key in keys_to_remove:
                        cls._log_files.pop(key, None)
                        cls._handlers.pop(key, None)

                except Exception as e:
                    print(f"Error removing log file {filename}: {e}")

    @classmethod
    def get_log_file_path(cls, module_name: str) -> Optional[str]:
        """Get the log file path for a specific module."""
        return cls._log_files.get(module_name)

    @classmethod
    def force_flush_all(cls):
        """Force flush all module handlers."""
        for handler in cls._handlers.values():
            try:
                handler.flush()
            except Exception:
                pass

    @classmethod
    def log_all_to_mlflow(cls, mlflow_handler) -> bool:
        """Log all individual log files as MLflow artifacts."""
        # Clean up unwanted log files first
        cls.cleanup_unwanted_logs(['main'])  # Only keep main.py logs

        # Force flush all handlers before uploading
        cls.force_flush_all()

        success_count = 0
        total_files = len(cls._log_files)

        for module_name, log_path in cls._log_files.items():
            if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
                try:
                    # Create clean module name for artifact path
                    clean_name = module_name.split('.')[-1] if '.' in module_name else module_name
                    if clean_name == '__main__':
                        clean_name = 'main'

                    # Upload to logs directory in MLflow
                    if mlflow_handler.log_artifact(log_path, "logs"):
                        success_count += 1
                        print(f"Successfully logged {clean_name} log file to MLflow")
                    else:
                        print(f"Failed to log {clean_name} log file to MLflow")
                except Exception as e:
                    print(f"Error logging {module_name} to MLflow: {e}")
            else:
                print(f"Log file not found or empty for {module_name}: {log_path}")

        return success_count == total_files

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def critical(self, msg):
        self.logger.critical(msg)

    def exception(self, msg):
        self.logger.exception(msg)