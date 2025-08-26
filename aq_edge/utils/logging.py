import logging
import os
from datetime import datetime

class LoggerHandler:
    def __init__(self, name=__name__, level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Only add handlers if they don't already exist
        if not self.logger.handlers:
            # Create artifacts directory if it doesn't exist
            artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'artifacts')
            os.makedirs(artifacts_dir, exist_ok=True)

            # Create log filename with timestamp and module name
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            module_name = name.split('.')[-1] if '.' in name else name
            log_filename = f"{module_name}_{timestamp}.log"
            log_path = os.path.join(artifacts_dir, log_filename)

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

            # Log the file location
            self.logger.info(f"Logging to file: {log_path}")

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def debug(self, msg):
        self.logger.debug(msg)
