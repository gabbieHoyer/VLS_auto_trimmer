
import subprocess
import argparse
import logging
import os

from src.utils import load_config, setup_logging

def run_pipeline(config_file):
    """Run the entire pipeline: process new data, retrain model, run inference, and handle failed inference."""
    # Common config file argument to pass to all subprocesses
    config_arg = ["--config_file", config_file]
    # config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", config_file)
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),  # Up three levels to root/
        "config",
        args.config_file
    )
    print(f"Config path: {config_path}")
    config = load_config(config_path)

    # Step 1: Process new data
    logging.info("Processing new data...")
    print("Processing new data...")
    subprocess.run(["python", "-m", "src.dataset_preprocessing.process_new_data"] + config_arg, check=True)

    # Step 2: Retrain model
    train_script = config["scripts"]["train_script"]
    logging.info(f"Retraining model using {train_script}...")
    print("Retraining model...")
    subprocess.run(["python", "-m", f"src.procedural_frame_classification.{train_script}"] + config_arg, check=True)

    # Step 3: Run inference
    inference_script = config["scripts"]["inference_script"]
    logging.info(f"Running inference using {inference_script}...")
    print("Running inference...")
    subprocess.run(["python", "-m", f"src.procedural_frame_classification.{inference_script}"] + config_arg, check=True)

    # Step 4: Handle failed inference
    # ** In Development:
    # logging.info("Handling failed inference...")
    # print("Handling failed inference...")
    # subprocess.run(["python", "-m", "src.dataset_preprocessing.handle_failed_inference"] + config_arg, check=True)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the full pipeline for data processing, training, inference, and handling failed inference.")
    parser.add_argument(
        "--config_file",
        default="config.yaml",
        help="Name of the config YAML file in the ../config/ folder (default: config.yaml)."
    )
    args = parser.parse_args()
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),  # Up three levels to root/
        "config",
        args.config_file
    )

    try:
        # Load config and set up logging
        config = load_config(config_path)
        logger = setup_logging(config["pipeline"]["log_file"], logger=logging.getLogger(__name__))

        run_pipeline(args.config_file)
        logger.info("Pipeline complete!")
        print("Pipeline complete!")

    except Exception as e:
        if 'logger' in locals():
            logger.error(f"An error occurred: {e}")
        else:
            print(f"An error occurred before logger was set up: {e}")
        raise

# python -m src.procedural_frame_classification.run_pipeline
# python -m src.procedural_frame_classification.run_pipeline --config_file config2.yaml
