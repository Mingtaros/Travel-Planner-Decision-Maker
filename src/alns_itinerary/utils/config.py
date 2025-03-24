"""
Configuration Handler
====================

This module provides utilities for loading and validating configuration settings
from JSON files. It gracefully handles errors and provides helpful error messages.

Usage:
    config = load_config("path/to/config.json")
    
    # Access configuration values
    api_key = config.get('api_key')
    max_locations = config.get('max_locations', 10)  # With default value
"""

import os
import json
import logging
import sys

logger = logging.getLogger("load_config")

def load_config(config_path="./src/alns_itinerary/config.json"):
    """
    Load and validate configuration settings from a JSON file.
    
    This function attempts to load the specified configuration file, providing
    helpful error messages if the file is missing, invalid, or inaccessible.
    The program will exit with status code 1 if configuration cannot be loaded.
    
    Args:
        config_path (str): Path to the JSON configuration file
                         (default: "./src/alns_itinerary/config.json")
    
    Returns:
        dict: Configuration parameters as a dictionary
    
    Raises:
        SystemExit: If the configuration file cannot be loaded
    
    Example:
        # Load with default path
        config = load_config()
        
        # Load with custom path
        config = load_config("./config/settings.json")
    """
    
    # Try to load configuration from file
    if not os.path.exists(config_path):
        logger.info(f"Error: Configuration file {config_path} not found.")
        logger.info("The program requires a valid configuration file to run.")
        sys.exit(1)
        
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        logger.info(f"Configuration loaded from {config_path}")
        return config
        
    except json.JSONDecodeError as e:
        logger.warning(f"Error parsing configuration file {config_path}: {e}")
        sys.exit(1)
    except IOError as e:
        logger.warning(f"Error reading configuration file {config_path}: {e}")
        sys.exit(1)