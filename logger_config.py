"""
Logger Configuration
Centralized logging for all modules
"""

import logging
import sys
from datetime import datetime

# Create logger
logger = logging.getLogger('FraudDetection')
logger.setLevel(logging.INFO)

# Create formatters
detailed_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

simple_formatter = logging.Formatter(
    '%(levelname)s - %(message)s'
)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(simple_formatter)

# File handler
log_filename = f'fraud_detection_{datetime.now().strftime("%Y%m%d")}.log'
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(detailed_formatter)

# Add handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Prevent duplicate logs
logger.propagate = False