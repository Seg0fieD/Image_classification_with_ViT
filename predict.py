import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
from datetime import datetime

import config
from model_ViT import create_vit_model
from utils import set_seed

def load_model(model_path=None):
    """
    Load a trained model.
    Args:
        model_path: Path to the model checkpoint, if None, loads the best model
    Returns:
        Loaded model
    """

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Create model
    model = create_vit_model().to(device)
    
    # Find best model if not specified
    if model_path is None:
        model_dir = config.MODEL_DIR
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory {model_dir} not found!")
        
        model_files = [f for f in os.listdir(model_dir) if f.startswith('best_model_')]
        if not model_files:
            raise FileNotFoundError("No model found in the model directory!")
        
        # Sort by F1 score to get the best model
        model_files.sort(key=lambda x: float(x.split('_')[-1].split('.')[0]))
        model_path = os.path.join(model_dir, model_files[-1])
        print(f"Using best model: {model_path}")
    
    # Load the model
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded with F1 score: {checkpoint.get('best_f1', 'N/A')}")
    return model, device

def get_test_transform():
    """Get the same transform as used for testing"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def process_image(image_path, transform=None):
    """
    Process an image for prediction.
    Args:
        image_path: Path to the image
        transform: Transform to apply to the image
    Returns:
        Transformed image tensor
    """
    if transform is None:
        transform = get_test_transform()
    
    # Open and convert the image
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None
    
    # Apply transformations
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor

def predict_image(model, image_tensor, device):
    """
    Make a prediction for a single image.
    Args:
        model: Trained model
        image_tensor: Processed image tensor
        device: Device to run inference on
    Returns:
        Predicted class index and probability
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_prob, predicted_idx = torch.max(probabilities, 1)
    
    return predicted_idx.item(), predicted_prob.item()

def predict_directory(model, directory, device, output_csv=None):
    """
    Make predictions for all images in a directory.
    Args:
        model: Trained model
        directory: Directory containing images
        device: Device to run inference on
        output_csv: Path to save results CSV
    Returns:
        DataFrame with predictions
    """
    import pandas as pd
    
    transform = get_test_transform()
    results = []
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print(f"No image files found in {directory}")
        return
    
    print(f"Found {len(image_files)} images to predict")
    
    # Process each image
    for image_path in image_files:
        img_tensor = process_image(image_path, transform)
        if img_tensor is None:
            continue
        
        pred_idx, pred_prob = predict_image(model, img_tensor, device)
        pred_class = config.CLASS_NAMES.get(pred_idx, f"Unknown ({pred_idx})")
        
        results.append({
            'filename': os.path.basename(image_path),
            'path': image_path,
            'predicted_class': pred_class,
            'confidence': pred_prob
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV if requested
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        results_df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
    
    return results_df

def visualize_prediction(image_path, predicted_class, confidence, save_path=None):
    """
    Visualize an image with its prediction.
    Args:
        image_path: Path to the image
        predicted_class: Predicted class name
        confidence: Prediction confidence
        save_path: Path to save the visualization
    """
    # Read the image
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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Make predictions with trained model')
    parser.add_argument('--model', type=str, default=None, help='Path to the model checkpoint')
    parser.add_argument('--image', type=str, default=None, help='Path to a single image')
    parser.add_argument('--dir', type=str, default=None, help='Path to directory of images')
    parser.add_argument('--output', type=str, default=None, help='Path to save output CSV')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Get current timestamp for filenames
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    
    # Check that at least one of image or directory is provided
    if args.image is None and args.dir is None:
        print("Error: Please provide either an image path or a directory of images")
        parser.print_help()
        return
    
    # Load model
    try:
        model, device = load_model(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Predict single image
    if args.image and not args.dir:
        if not os.path.exists(args.image):
            print(f"Error: Image {args.image} not found")
            return
        
        print(f"Predicting image: {args.image}")
        img_tensor = process_image(args.image)
        if img_tensor is not None:
            pred_idx, pred_prob = predict_image(model, img_tensor, device)
            pred_class = config.CLASS_NAMES.get(pred_idx, f"Unknown ({pred_idx})")
            
            print(f"Prediction: {pred_class}")
            print(f"Confidence: {pred_prob:.4f}")
            
            # Visualize if requested
            if args.visualize:
                # Create the predictions directory if it doesn't exist
                os.makedirs(config.PRED_DIR_OUT, exist_ok=True)
                
                # Generate the filename with timestamp
                viz_path = os.path.join(config.PRED_DIR_OUT, f"prediction_{timestamp}.png")
                
                # Use custom output path if provided
                if args.output:
                    viz_path = args.output
                
                # Call visualization function with an explicit path
                visualize_prediction(args.image, pred_class, pred_prob, viz_path)
                print(f"Visualization saved to {viz_path}")
    
    # Predict directory of images
    if args.dir:
        if not os.path.exists(args.dir):
            print(f"Error: Directory {args.dir} not found")
            return
        
        # Create timestamped subdirectory for predictions
        prediction_subdir = os.path.join(config.PRED_DIR_OUT, timestamp)
        os.makedirs(prediction_subdir, exist_ok=True)
        print(f"Created prediction directory: {prediction_subdir}")
        
        # Set CSV output path
        output_csv = args.output
        if output_csv is None:
            output_csv = os.path.join(prediction_subdir, 'predictions.csv')
        
        print(f"Predicting images in directory: {args.dir}")
        results_df = predict_directory(model, args.dir, device, output_csv)
        
        # Display results summary
        if results_df is not None and not results_df.empty:
            print("\nPrediction Summary:")
            counts = results_df['predicted_class'].value_counts()
            for cls, count in counts.items():
                print(f"  {cls}: {count} images")
            
            # Visualize predictions if requested
            if args.visualize:
                # Process all images, not just 5
                for idx, row in results_df.iterrows():
                    viz_path = os.path.join(prediction_subdir, f"{os.path.splitext(row['filename'])[0]}_prediction.png")
                    visualize_prediction(row['path'], row['predicted_class'], row['confidence'], viz_path)
                print(f"Visualizations saved to {prediction_subdir}")
if __name__ == "__main__":
    main()