import logging
import sys
from datetime import datetime
import os
class Logger:
    @staticmethod
    def setup_logger(name, log_file=None, level=logging.INFO):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        return logger
    
    def get_project_logger():
        """Получение логгера для проекта"""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"voice_detector_{timestamp}.log")
        return Logger.setup_logger("VoiceDetector", log_file=log_file)