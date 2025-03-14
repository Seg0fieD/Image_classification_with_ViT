# Intel Image Classification using Vision Transformer

This project implements a Vision Transformer (ViT) based approach for classifying natural scenes from the Intel Image Classification dataset.

## Dataset Overview

The Intel Image Classification dataset contains around 25,000 images (150x150) distributed across 6 categories:
- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

Dataset splits: training (~14,000 images), testing (~3,000 images), prediction (~7,000 images)

## Project directory structure

```
Image_classification_task/
├── data/                                   # Dataset directory
│   ├── seg_train/                          # Training images by class folders
│   ├── seg_test/                           # Test images by class folders
│   ├── seg_pred/                           # Prediction images 
│   ├── demo1/                              # Custom images for demonstration
│   ├── demo2/                              # Additional demo images
│   ├── single_img1                         # Images for single prediction demonstration
│   └── single_img2                         
├── model_ViT.py                            # Vision Transformer model definition
├── train.py                                # Training pipeline
├── dataset.py                              # Data loading and preprocessing
├── predict.py                              # Inference script for new images
├── utils.py                                # Utility functions
├── config.py                               # Configuration parameters
├── main.py                                 
└── output/                                 # Outputs directory
    ├── plots/                              # Plots and visualizations
    ├── predictions/                        # Model predictions visualizations
    │   ├── 2025-03-14_071510/              # Predictions from demo1 data
    │   ├── 2025-03-14_074838_(seg_pred)/   # First 100 images from prediction dataset
    │   ├── 2025-03-14_081944/              # Predictions from demo2 data 
    │   ├── prediction_20250314_173131.png  # Single images inference for demonstration
    │   └── prediction_20250314_173211.png  
    ├── logs/                               # Training logs
    ├── csv/                                
    └── best_model/                         # Saved model checkpoints
```

## Model Architecture

The model uses pre-trained Vision Transformer (ViT-B/16) with a custom classification head:

```python
model = timm.create_model("vit_base_patch16_224", pretrained=True)
model.head = nn.Sequential(
    nn.Linear(model.head.in_features, 512),
    nn.GELU(),
    nn.Dropout(0.3),
    nn.Linear(512, 6)  # 6 classes
)
```

## Training Process

The training can be done in either two-stage (head-only, then fine-tuning) or single-stage approach. Training uses Cross-Entropy Loss with AdamW optimizer, Warm-up + Cosine Annealing scheduling, and early stopping based on F1 score.

## Evaluation Metrics

- Accuracy
- F1 Score (weighted, used for model selection)
- Precision
- Recall

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

```bash
# Single image
python predict.py --image data/single_img1 --visualize

# Directory of images
python predict.py --dir data/demo2 --visualize

# Using specific model
python predict.py --model output/best_model/best_model_0.9177.pth --image data/single_img1 --output output/predictions
```

## Dependencies

PyTorch, torchvision, timm, tensorboard, scikit-learn, pandas, matplotlib, seaborn, opencv-python, tqdm

## Acknowledgements

- Dataset originally published by Intel for the "Intel Image Classification Challenge" on Analytics Vidhya
- Vision Transformer implementation based on the timm library by Ross Wightman