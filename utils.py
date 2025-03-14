import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
from torch.utils.data import Sampler
import logging
from datetime import datetime
import config

def setup_logger():
    """Set up logger to save logs to a file"""
    logger = logging.getLogger('image_classification')
    logger.setLevel(logging.INFO)
    
    c_handler = logging.StreamHandler()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.LOG_DIR, f'training_{timestamp}.log')
    f_handler = logging.FileHandler(log_file)
    
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(log_format)
    f_handler.setFormatter(log_format)
    
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

    indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
    fig, axes = plt.subplots(1, len(indices), figsize=(15, 3))
    
    for i, idx in enumerate(indices):
        img = images[idx].permute(1, 2, 0).cpu().numpy()
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

def visualize_prediction(image_path, predicted_class, confidence, save_path=None, verbose=True):
    """Visualize an image with its prediction."""
    # Read the image
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Convert from BGR to RGB for matplotlib
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Display the image with prediction
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_class} (Confidence: {confidence:.2f})")
    plt.axis('off')
    
    if save_path:
        if os.path.isdir(save_path):
            from datetime import datetime
            filename = f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            save_path = os.path.join(save_path, filename)
        
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        plt.savefig(save_path)
        if verbose:
            print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_prediction_distribution(predictions, save_path):
    """
    Create a bar plot showing distribution of predicted classes
    Args:
        predictions: DataFrame containing prediction results
        save_path: Path to save the plot
    """
    if predictions is None or len(predictions) == 0:
        print("No prediction data available for plotting")
        return
        
    # Count class frequencies
    class_counts = predictions['predicted_class'].value_counts()
    
    plt.figure(figsize=(10, 6))
    class_counts.plot(kind='bar', color='skyblue')
    plt.title('Distribution of Predicted Classes')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path)
    plt.close()
    print(f"Class distribution plot saved to {save_path}")

def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    return accuracy, f1, precision, recall

def log_metrics_to_tensorboard(writer, metrics, epoch, prefix=''):
    """Log metrics to TensorBoard"""
    train_loss, val_loss, accuracy, f1 = metrics
    writer.add_scalar(f'{prefix}train_loss', train_loss, epoch)
    writer.add_scalar(f'{prefix}val_loss', val_loss, epoch)
    writer.add_scalar(f'{prefix}accuracy', accuracy, epoch)
    writer.add_scalar(f'{prefix}f1', f1, epoch)