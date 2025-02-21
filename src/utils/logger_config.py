import logging
import os
from datetime import datetime

def setup_logger(name):
    """
    Basic Custom Logging formatting and handling

    Parameters
    name (str) : Name of the logger 

    Returns:
    logging.Logger : Configured Logger Instance
    """

    os.makedirs('logs',exist_ok=True)

    # Create Logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Configs for how logs will appear in logs/
    file_format = logging.Formatter(
        '%(levelname)s : %(name)s : %(funcName)s : %(lineno)d : %(message)s'
    )

    log_file = f'logs/chicago_crime_{datetime.now().strftime("%m%d%Y")}'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_format)

    # Console logs configs
    console_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(console_format)


    # Add the config to the Logger obj
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger





