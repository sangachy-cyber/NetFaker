#!/usr/bin/env python3
"""HoloWAN training dataset generation script.

Entry point for dataset generation pipeline.
Responsible for:
1. Command line interface
2. Calling train_test_splitter.generate_datasets()
3. Printing statistics and logs
4. No business logic - only glue code

Usage:
    # Generate datasets with default parameters
    uv run python scripts/generate_training_dataset.py

    # Specify custom directories
    uv run python scripts/generate_training_dataset.py --processed-data-dir=data/processed/ --datasets-dir=data/datasets/
"""

import argparse
import logging
import os

from netfaker.simcore.dataset.train_test_splitter import TrainTestSplitter


def main():
    """Main function for dataset generation."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate HoloWAN training datasets')

    parser.add_argument(
        '--processed-data-dir',
        type=str,
        default='data/processed/',
        help='Directory containing processed Parquet files (default: data/processed/)'
    )

    parser.add_argument(
        '--datasets-dir',
        type=str,
        default='data/datasets/',
        help='Directory to save generated datasets (default: data/datasets/)'
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.exists(args.processed_data_dir):
        logging.error("Processed data directory does not exist: %s", args.processed_data_dir)
        return 1

    # Create output directory if it doesn't exist
    os.makedirs(args.datasets_dir, exist_ok=True)

    # Initialize splitter and generate datasets
    splitter = TrainTestSplitter(
        processed_data_dir=args.processed_data_dir,
        datasets_dir=args.datasets_dir
    )

    try:
        logging.info("=== Starting HoloWAN dataset generation ===")
        logging.info("Input directory: %s", args.processed_data_dir)
        logging.info("Output directory: %s", args.datasets_dir)
        logging.info("")

        # Generate datasets
        logging.info("Calling train_test_splitter.generate_datasets()...")
        train_size, test_size = splitter.generate_datasets()

        logging.info("")
        logging.info("=== HoloWAN dataset generation completed ===")
        logging.info("Total train samples: %d", train_size)
        logging.info("Total test samples: %d", test_size)
        logging.info("Output files:")
        logging.info("  - %s/train.parquet", args.datasets_dir)
        logging.info("  - %s/test.parquet", args.datasets_dir)

        if train_size == 0 and test_size == 0:
            logging.warning("No samples generated. Check if input files are valid.")
            return 1

        return 0

    except Exception as e:
        logging.error("Error during dataset generation: %s", str(e))
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
