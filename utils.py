import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from torch.utils.data import Sampler
import logging
from datetime import datetime
import config

def setup_logger():
    """Set up logger to save logs to a file"""
    logger = logging.getLogger('image_classification')
    logger.setLevel(logging.INFO)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.LOG_DIR, f'training_{timestamp}.log')
    f_handler = logging.FileHandler(log_file)
    
    # Create formatters and add to handlers
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(log_format)
    f_handler.setFormatter(log_format)
    
    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    return logger

def set_seed(seed):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, 'mps'):
        torch.mps.manual_seed(seed)
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    """Function to ensure seed for dataloader workers"""
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class CustomRandomSampler(Sampler):
    """Custom sampler for dataloader"""
    def __init__(self, data_source):
        self.data_source = data_source
        
    def __iter__(self):
        indices = list(range(len(self.data_source)))
        random.shuffle(indices)
        return iter(indices)
        
    def __len__(self):
        return len(self.data_source)

def count_images(directory):
    """Count images in each category"""
    count_class = {}
    total_img = 0
    if os.path.exists(directory):
        for class_name in os.listdir(directory):
            class_path = os.path.join(directory, class_name)
            if os.path.isdir(class_path):
                img_count = len(os.listdir(class_path))
                count_class[class_name] = img_count
                total_img += img_count
    return total_img, count_class

def plot_dataset_distribution(train_count, test_count, save_path):
    """Plot the distribution of classes in train and test datasets"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Bar plot for Train Dataset
    axes[0].bar(train_count.keys(), train_count.values(), color='skyblue')
    axes[0].set_title('Train Dataset Distribution')
    axes[0].set_xlabel('Classes')
    axes[0].set_ylabel('Number of Images')
    
    # Bar plot for Test Dataset
    axes[1].bar(test_count.keys(), test_count.values(), color='lightgreen')
    axes[1].set_title('Test Dataset Distribution')
    axes[1].set_xlabel('Classes')
    axes[1].set_ylabel('Number of Images')
    
    # Total dataset size comparison
    all_img_count = {"Train": sum(train_count.values()), "Test": sum(test_count.values())}
    axes[2].bar(all_img_count.keys(), all_img_count.values(), color='lightcoral')
    axes[2].set_title('Total Dataset Distribution')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_learning_curves(train_losses, val_losses, val_f1s, val_accs, save_dir):
    """Plot learning curves for training and validation"""
    # Plot training and validation loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation metrics
    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label='Accuracy')
    plt.plot(val_f1s, label='F1 Score')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curves.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_predictions(images, true_labels, pred_labels, class_names, save_path, num_samples=5):
    """Visualize sample predictions"""
    # Select a subset of samples
    indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
    
    fig, axes = plt.subplots(1, len(indices), figsize=(15, 3))
    
    for i, idx in enumerate(indices):
        # Convert tensor to image
        img = images[idx].permute(1, 2, 0).cpu().numpy()
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f"True: {class_names[true_labels[idx]]}\nPred: {class_names[pred_labels[idx]]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_predictions_to_csv(filenames, predictions, save_path):
    """Save predictions to CSV file"""
    df = pd.DataFrame({
        'Filename': filenames,
        'Category': [config.CLASS_NAMES[pred] for pred in predictions]
    })
    df.to_csv(save_path, index=False)
    return df

def get_class_distribution(predictions, save_path):
    """Plot distribution of predicted classes"""
    plt.figure(figsize=(10, 6))
    
    # Convert numerical predictions to class names
    if isinstance(predictions[0], int):
        class_predictions = [config.CLASS_NAMES[pred] for pred in predictions]
    else:
        class_predictions = predictions
        
    # Count occurrences of each class
    class_counts = {}
    for cls in class_predictions:
        if cls in class_counts:
            class_counts[cls] += 1
        else:
            class_counts[cls] = 1
    
    # Plot
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title('Distribution of Predicted Classes')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return class_counts