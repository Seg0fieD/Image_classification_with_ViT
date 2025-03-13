import argparse
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from train import train_model
import utils

def main():
    parser = argparse.ArgumentParser(description='Intel Image Classification using Vision Transformer')
    parser.add_argument('--seed', type=int, default=config.SEED, help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE, help='Learning rate')
    parser.add_argument('--single_stage', action='store_true', help='Use single stage training instead of two-stage')
    parser.add_argument('--workers', type=int, default=config.NUM_WORKERS, help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config.SEED = args.seed
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    config.NUM_WORKERS = args.workers
    
    # Set up seed
    utils.set_seed(config.SEED)
    
    # Create output directories
    for dir_path in [config.OUTPUT_DIR, config.PLOT_DIR, config.LOG_DIR, 
                     config.CSV_DIR, config.PRED_DIR_OUT, config.MODEL_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Start training
    train_model(train_two_stage=not args.single_stage)

if __name__ == "__main__":
    main()