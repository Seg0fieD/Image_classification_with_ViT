import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import config
from utils import CustomRandomSampler, worker_init_fn

def get_transforms():
    """Get data transformations for train and test datasets"""
    # Data augmentation for train dataset
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=35),
        transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.5, 1.5)),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Transformations for test dataset
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform

def load_datasets():
    """Load train and test datasets"""
    train_transform, test_transform = get_transforms()
    
    # Load train and test datasets
    train_dataset = datasets.ImageFolder(root=config.TRAIN_DIR, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=config.TEST_DIR, transform=test_transform)
    
    return train_dataset, test_dataset

def create_dataloaders(train_dataset, test_dataset):
    """Create dataloaders for train and test datasets"""
    # Create train sampler
    train_sampler = CustomRandomSampler(train_dataset)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        sampler=train_sampler, 
        num_workers=config.NUM_WORKERS,
        worker_init_fn=worker_init_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    return train_loader, test_loader