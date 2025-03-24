import os
import json
import logging
import sys

logger = logging.getLogger("load_config")

def load_config(config_path="./src/alns_itinerary/config.json"):
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        dict: Configuration parameters
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