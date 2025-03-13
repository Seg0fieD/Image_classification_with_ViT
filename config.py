import os

# Paths
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "seg_train/seg_train")
TEST_DIR = os.path.join(DATA_DIR, "seg_test/seg_test")
PRED_DIR = os.path.join(DATA_DIR, "seg_pred")

# Output directories
OUTPUT_DIR = "output"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
CSV_DIR = os.path.join(OUTPUT_DIR, "csv")
PRED_DIR_OUT = os.path.join(OUTPUT_DIR, "predictions")
MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model")

# Create output directories if they don't exist
for dir_path in [OUTPUT_DIR, PLOT_DIR, LOG_DIR, CSV_DIR, PRED_DIR_OUT, MODEL_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Training parameters
BATCH_SIZE = 16
NUM_WORKERS = 4
SEED = 37
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
PATIENCE = 3  # Early stopping patience

# Model parameters
NUM_CLASSES = 6
FREEZE_BASE_MODEL = True

# Class names
CLASS_NAMES = {
    0: 'buildings',
    1: 'forest',
    2: 'glacier',
    3: 'mountain',
    4: 'sea',
    5: 'street'
}

CLASS_IDX = {value: key for key, value in CLASS_NAMES.items()}