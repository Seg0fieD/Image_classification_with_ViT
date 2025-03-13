import torch
import torch.nn as nn
import timm
import config

def create_vit_model(num_classes=config.NUM_CLASSES, pretrained=True):
    """Create a Vision Transformer model with a custom classification head"""
    # Load pretrained ViT-B_16 model
    model = timm.create_model("vit_base_patch16_224", pretrained=pretrained)
    
    # Modify the classification head
    num_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model

def freeze_base_model(model):
    """Freeze all layers except the head"""
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze head layers
    for param in model.head.parameters():
        param.requires_grad = True
    
    return model

def unfreeze_last_blocks(model, num_blocks=3):
    """Unfreeze the last few transformer blocks for fine-tuning"""
    # Unfreeze specified number of last blocks
    for i, block in enumerate(model.blocks):
        if i >= len(model.blocks) - num_blocks:
            for param in block.parameters():
                param.requires_grad = True
    
    return model

def get_optimizer(model, lr=config.LEARNING_RATE, stage='head_only'):
    """Get optimizer based on training stage"""
    if stage == 'head_only':
        # Only optimize the head
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr)
    elif stage == 'fine_tuning':
        # Different learning rates for different parts
        trainable_params = [
            {'params': model.head.parameters(), 'lr': lr},
            {'params': [p for i, block in enumerate(model.blocks) 
                         if i >= len(model.blocks) - 3 
                         for p in block.parameters()], 'lr': lr/10}
        ]
        optimizer = torch.optim.AdamW(trainable_params)
    else:
        # Train all parameters
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    return optimizer

def get_scheduler(optimizer, train_loader, num_epochs):
    """Create a learning rate scheduler"""
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    
    # Warm-up phase
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.1, 
        end_factor=1.0, 
        total_iters=len(train_loader)
    )
    
    # Main scheduler
    main_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs*len(train_loader)
    )
    
    # Combined scheduler
    scheduler = SequentialLR(
        optimizer, 
        [warmup_scheduler, main_scheduler], 
        [len(train_loader)]
    )
    
    return scheduler