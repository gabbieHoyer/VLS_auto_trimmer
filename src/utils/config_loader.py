import os
import sys
import yaml
import logging
from pathlib import Path

def convert_to_float(value):
    """Recursively convert scientific notation strings to floats."""
    if isinstance(value, str):
        try:
            # Try to convert the string to a float
            return float(value)
        except ValueError:
            # If conversion fails, return the original string
            return value
    elif isinstance(value, dict):
        # Recursively process dictionaries
        return {k: convert_to_float(v) for k, v in value.items()}
    elif isinstance(value, list):
        # Recursively process lists
        return [convert_to_float(item) for item in value]
    else:
        # Return non-string values unchanged
        return value

def resolve_path(path, base_dir):
    """Resolve a path relative to the base directory if it's not absolute."""
    if path and not os.path.isabs(path):
        return os.path.abspath(os.path.join(base_dir, path))
    return path

def load_config(config_path="config.yaml"):
    """Load configuration from a YAML file, convert scientific notation strings to floats, and resolve relative paths."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Recursively convert scientific notation strings to floats
    config = convert_to_float(config)
    
    # Get the directory of the config file
    config_dir = os.path.dirname(config_path)
    
    # Resolve paths in 'paths' section
    if "paths" in config:
        for key in config["paths"]:
            config["paths"][key] = resolve_path(config["paths"][key], config_dir)
    
    # Resolve paths in 'pipeline' section
    if "pipeline" in config and "log_file" in config["pipeline"]:
        config["pipeline"]["log_file"] = resolve_path(config["pipeline"]["log_file"], config_dir)
    
    # Resolve paths in 'inference' section
    if "inference" in config:
        # Do NOT resolve weights_path here; let scripts handle it relative to output_base_dir
        if "inference_output_dir" in config["inference"]:
            config["inference"]["inference_output_dir"] = resolve_path(config["inference"]["inference_output_dir"], config_dir)
    
    # Resolve paths in 'auto_trim' section
    if "auto_trim" in config:
        if "input_path" in config["auto_trim"]:
            config["auto_trim"]["input_path"] = resolve_path(config["auto_trim"]["input_path"], config_dir)
        if "csv_path" in config["auto_trim"]:
            config["auto_trim"]["csv_path"] = resolve_path(config["auto_trim"]["csv_path"], config_dir)
    
    return config

def load_config_lite(config_path="config.yaml"):
    """Load configuration from a YAML file and convert scientific notation strings to floats."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # Recursively convert scientific notation strings to floats
    config = convert_to_float(config)
    return config

def setup_logging(log_file, logger):
    """Set up logging to write to the specified log file."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)  # Ensure log directory exists
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Also log to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(console_handler)

    return logger
