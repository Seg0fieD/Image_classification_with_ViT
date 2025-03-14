import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter
import logging

import config
import utils
from model_ViT import create_vit_model, freeze_base_model, unfreeze_last_blocks, get_optimizer, get_scheduler
from dataset import load_datasets, create_dataloaders

def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, logger):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    
    for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        if device.type == "cuda":
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
        elif device.type == "mps":
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    logger.info(f"Training Loss: {avg_train_loss:.4f}")
    
    return avg_train_loss

def evaluate(model, test_loader, criterion, device, logger):
    """Evaluate the model"""
    model.eval()
    all_preds = []
    all_labels = []
    val_loss = 0.0
    all_images = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if len(all_images) < 10:
                all_images.extend(inputs[:min(5, len(inputs))])
    
    # Calculate metrics 
    accuracy, f1, precision, recall = utils.calculate_metrics(all_labels, all_preds)
    avg_val_loss = val_loss / len(test_loader)
    logger.info(f"Validation: Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.4f} | F1: {f1:.4f}")
    
    return avg_val_loss, accuracy, f1, precision, recall, all_images[:5], all_labels[:5], all_preds[:5]

def save_best_model(model, optimizer, scheduler, epoch, best_f1, model_dir):
    """Save best model checkpoint"""
    model_path = os.path.join(model_dir, f'best_model_{best_f1:.4f}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_f1': best_f1,
    }, model_path)
    return model_path

def train_model(train_two_stage=True):
    """Train the model with option for two-stage training"""
   
    config.validate_config()

    utils.set_seed(config.SEED)
    logger = utils.setup_logger()
    
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    writer = SummaryWriter(log_dir=config.LOG_DIR)
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset, test_dataset = load_datasets()
    train_loader, test_loader = create_dataloaders(train_dataset, test_dataset)
    
    # Plot dataset distribution
    train_count_total, train_count = utils.count_images(config.TRAIN_DIR)
    test_count_total, test_count = utils.count_images(config.TEST_DIR)
    utils.plot_dataset_distribution(
        train_count, 
        test_count, 
        os.path.join(config.PLOT_DIR, 'dataset_distribution.png')
    )
    logger.info(f"Train: {train_count_total} images, Test: {test_count_total} images")
    
    # Model
    logger.info("Creating model...")
    model = create_vit_model().to(device)
    
    # Loss 
    criterion = nn.CrossEntropyLoss()
    
    # Training metrics
    train_losses = []
    val_losses = []
    val_accs = []
    val_f1s = []
    best_f1 = 0
    patience_counter = 0
    
    if train_two_stage:
        # Stage 1: Train only the head
        logger.info("Stage 1: <<<< Training only the head >>>>")
        model = freeze_base_model(model)
        optimizer = get_optimizer(model, stage='head_only')
        scheduler = get_scheduler(optimizer, train_loader, config.NUM_EPOCHS//2)
        
        for epoch in range(config.NUM_EPOCHS//2):
            logger.info(f"Epoch {epoch+1}/{config.NUM_EPOCHS//2}")
            
            # Train
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, logger)
            train_losses.append(train_loss)
            
            # Evaluate
            val_loss, accuracy, f1, precision, recall, val_images, val_labels, val_preds = evaluate(
                model, test_loader, criterion, device, logger
            )
            val_losses.append(val_loss)
            val_accs.append(accuracy)
            val_f1s.append(f1)
            
            utils.log_metrics_to_tensorboard(writer, [train_loss, val_loss, accuracy, f1], epoch, prefix='stage1_')
            
            # Check improvement
            if f1 > best_f1:
                best_f1 = f1
                patience_counter = 0
                model_path = save_best_model(model, optimizer, scheduler, epoch, best_f1, config.MODEL_DIR)
                logger.info(f"New best model saved with F1: {best_f1:.4f}")
                
                # Visualize sample predictions
                utils.visualize_predictions(
                    val_images, 
                    val_labels, 
                    val_preds, 
                    config.CLASS_NAMES, 
                    os.path.join(config.PLOT_DIR, f'sample_predictions_stage1_epoch{epoch+1}.png')
                )
            else:
                patience_counter += 1
                if patience_counter >= config.PATIENCE:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Stage 2: Fine-tune the last few blocks
        logger.info("Stage 2: <<<< Fine-tuning the last transformer blocks >>>>")
        model = unfreeze_last_blocks(model)
        optimizer = get_optimizer(model, stage='fine_tuning')
        scheduler = get_scheduler(optimizer, train_loader, config.NUM_EPOCHS - config.NUM_EPOCHS//2)
        patience_counter = 0
        
        for epoch in range(config.NUM_EPOCHS - config.NUM_EPOCHS//2):
            logger.info(f"Epoch {epoch+1}/{config.NUM_EPOCHS - config.NUM_EPOCHS//2}")
            

            train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, logger)
            train_losses.append(train_loss)
            

            val_loss, accuracy, f1, precision, recall, val_images, val_labels, val_preds = evaluate(
                model, test_loader, criterion, device, logger
            )
            val_losses.append(val_loss)
            val_accs.append(accuracy)
            val_f1s.append(f1)

            utils.log_metrics_to_tensorboard(writer, [train_loss, val_loss, accuracy, f1], 
                                             epoch + config.NUM_EPOCHS//2, prefix='stage2_')
            
            # Check improvement
            if f1 > best_f1:
                best_f1 = f1
                patience_counter = 0
                model_path = save_best_model(model, optimizer, scheduler, 
                                             epoch + config.NUM_EPOCHS//2, best_f1, config.MODEL_DIR)
                logger.info(f"New best model saved with F1: {best_f1:.4f}")
                
                # Visualize sample predictions
                utils.visualize_predictions(
                    val_images, 
                    val_labels, 
                    val_preds, 
                    config.CLASS_NAMES, 
                    os.path.join(config.PLOT_DIR, f'sample_predictions_stage2_epoch{epoch+1}.png')
                )
            else:
                patience_counter += 1
                if patience_counter >= config.PATIENCE:
                    logger.info(f"Early stopping at epoch {epoch+1} of stage 2")
                    break
    else:
        # Single stage training
        optimizer = get_optimizer(model, stage='all')
        scheduler = get_scheduler(optimizer, train_loader, config.NUM_EPOCHS)
        
        for epoch in range(config.NUM_EPOCHS):
            logger.info(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
            
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, logger)
            train_losses.append(train_loss)
            
            val_loss, accuracy, f1, precision, recall, val_images, val_labels, val_preds = evaluate(
                model, test_loader, criterion, device, logger
            )
            val_losses.append(val_loss)
            val_accs.append(accuracy)
            val_f1s.append(f1)
            
            utils.log_metrics_to_tensorboard(writer, [train_loss, val_loss, accuracy, f1], epoch)
            
            if f1 > best_f1:
                best_f1 = f1
                patience_counter = 0
                model_path = save_best_model(model, optimizer, scheduler, epoch, best_f1, config.MODEL_DIR)
                logger.info(f"New best model saved with F1: {best_f1:.4f}")
                
                utils.visualize_predictions(
                    val_images, 
                    val_labels, 
                    val_preds, 
                    config.CLASS_NAMES, 
                    os.path.join(config.PLOT_DIR, f'sample_predictions_epoch{epoch+1}.png')
                )
            else:
                patience_counter += 1
                if patience_counter >= config.PATIENCE:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
    
    # Plot learning curves
    utils.plot_learning_curves(
        train_losses,
        val_losses,
        val_f1s,
        val_accs,
        config.PLOT_DIR
    )
    
    # Load best model 
    logger.info("Loading best model for final evaluation...")
    best_model_path = sorted([f for f in os.listdir(config.MODEL_DIR) if f.startswith('best_model_')])[-1]
    best_model_path = os.path.join(config.MODEL_DIR, best_model_path)
    
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded best model with F1: {checkpoint['best_f1']:.4f}")
    
    # Final evaluation on test set
    val_loss, accuracy, f1, precision, recall, val_images, val_labels, val_preds = evaluate(
        model, test_loader, criterion, device, logger
    )
    
    # Create Classification report
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
    
    class_names = list(config.CLASS_NAMES.values())
    report = metrics.classification_report(y_true, y_pred, target_names=class_names)
    logger.info("\nClassification Report:\n" + report)
    
    with open(os.path.join(config.LOG_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    utils.plot_confusion_matrix(
        y_true, 
        y_pred, 
        class_names, 
        os.path.join(config.PLOT_DIR, 'confusion_matrix.png')
    )
    
    writer.close()
    logger.info("Training completed!")
    
    return best_f1

if __name__ == "__main__":
    train_model(train_two_stage=True)