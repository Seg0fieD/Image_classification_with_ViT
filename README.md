# Image_classification_with_ViT

## Intel Image Classification using Vision Transformer

This project implements a Vision Transformer (ViT) based approach for classifying natural scenes from the Intel Image Classification dataset.

## Dataset Overview

The Intel Image Classification dataset contains around 25,000 images (150x150) distributed across 6 categories:
- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

The dataset is divided into:
- Training set: ~14,000 images
- Testing set: ~3,000 images
- Prediction set: ~7,000 images

## Project Structure

```
Image_classification_task/
├── data/                  # Dataset directory
│   ├── seg_train/         # Training images organized by class folders
│   ├── seg_test/          # Test images organized by class folders
│   ├── seg_pred/          # Prediction images 
│   └── demo_/             # Custom images for demonstration
├── model_ViT.py           # Vision Transformer model definition
├── train.py               # Training pipeline
├── dataset.py             # Data loading and preprocessing
├── predict.py             # Inference script for new images
├── utils.py               # Utility functions
├── config.py              # Configuration parameters
├── main.py                # Entry point script
└── output/                # Generated outputs
    ├── plots/             # Training plots and visualizations
    ├── predictions/       # Model predictions visualizations
    ├── logs/              # Training logs
    ├── csv/               # Prediction results in CSV format
    └── best_model/        # Saved model checkpoints
```

## Model Architecture

The model is based on the pre-trained Vision Transformer (ViT-B/16) with a custom classification head added for the specific task:

```python
model = timm.create_model("vit_base_patch16_224", pretrained=True)
num_features = model.head.in_features
model.head = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.GELU(),
    nn.Dropout(0.3),
    nn.Linear(512, 6)  # 6 classes
)
```

This architecture:
1. Uses a pre-trained ViT-B/16 as the backbone
2. Replaces the classification head with a custom network
3. Adds a hidden layer with GELU activation and dropout for regularization
4. Outputs predictions for the 6 classes

## Training Process

The training follows a two-stage approach:

### Stage 1: Head-only Training
- Freeze the backbone (all ViT layers)
- Train only the custom classification head
- This allows the model to quickly adapt to the new classification task

### Stage 2: Fine-tuning
- Unfreeze the last transformer blocks
- Fine-tune these layers with a lower learning rate
- This allows the model to adapt its feature representations to the specific dataset

### Training Parameters
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: AdamW
- **Learning Rate Scheduler**: Warm-up followed by Cosine Annealing
- **Metrics**: 
  - Accuracy
  - F1 Score (weighted, used for model selection)
  - Precision
  - Recall
- **Early Stopping**: Based on F1 score with patience of 3 epochs
- **Mixed Precision Training**: Used for faster training on compatible GPUs

## Results

The model achieves strong performance on the validation set:
- F1 Score: >0.90
- Accuracy: >0.90

Performance varies by class, with some natural scenes being easier to distinguish than others. The model is particularly effective at distinguishing forests and glaciers, while occasionally confusing buildings and streets.

## Usage

### Training

```bash
# Train with default parameters (two-stage approach)
python train.py

# Train with custom parameters
python main.py --epochs 10 --batch_size 16 --lr 0.001 

# Train with single-stage approach
python main.py --single_stage
```

### Prediction

For a single image:
```bash
python predict.py --image data/img1.jpeg --visualize
```

For a directory of images:
```bash
python predict.py --dir data/demo_ --visualize
```

For using a specific model:
```bash
python predict.py --model output/best_model/best_model_0.9177.pth --image data/img1.jpeg
```

## Dependencies

- PyTorch
- torchvision
- timm
- tensorboard
- scikit-learn
- pandas
- matplotlib
- seaborn
- opencv-python
- tqdm

## Installation

```bash
pip install torch torchvision tqdm matplotlib seaborn scikit-learn pandas timm tensorboard opencv-python
```

## Acknowledgements

- The dataset was originally published by Intel for the "Intel Image Classification Challenge" on Analytics Vidhya
- The Vision Transformer implementation is based on the timm library by Ross Wightman
