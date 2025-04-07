import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.unet import UNet
from utils.dataset import JSRTDataset
from config import TrainingConfig

def create_transforms(is_train=True):
    """Create transforms for training and validation"""
    if is_train:
        return A.Compose([
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
            ], p=0.3),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(p=0.5),
            ], p=0.3),
            A.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                translate_percent={"x": (-0.0625, 0.0625), "y": (-0.0625, 0.0625)},
                rotate=(-45, 45),
                p=0.5
            ),
            A.Normalize(mean=[0.485], std=[0.229]),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})
    else:
        return A.Compose([
            A.Normalize(mean=[0.485], std=[0.229]),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})

def dice_loss(pred, target, smooth=1e-5):
    """
    Calculate Dice loss with better numerical stability
    """
    # Apply sigmoid activation
    pred = torch.sigmoid(pred)
    
    # Flatten the predictions and targets
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Calculate intersection and union
    intersection = (pred * target).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()
    
    # Calculate dice coefficient
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    
    # Return loss
    return 1.0 - dice

def dice_score(pred, target, smooth=1e-8):
    """
    Calculate Dice score (metric)
    """
    # Apply threshold to get binary predictions
    pred = (pred > 0.5).float()
    
    # Flatten the predictions and targets
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Calculate intersection and union
    intersection = (pred * target).sum()
    denominator = pred.sum() + target.sum()
    
    # Add small epsilon to avoid division by zero
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    
    return dice.item()  # Convert to Python scalar

def train_epoch(model, train_loader, optimizer, criterion, device, gradient_clip=None):
    """Train one epoch"""
    model.train()
    total_loss = 0
    total_dice = 0
    num_batches = len(train_loader)
    
    try:
        pbar = tqdm(train_loader, desc='Training', ncols=100)
        for batch_idx, (images, masks) in enumerate(pbar):
            try:
                # Move to device
                images = images.to(device)
                masks = masks.to(device)
                
                # Debug info for first batch of first epoch
                if batch_idx == 0 and not hasattr(train_epoch, 'first_batch_done'):
                    print(f"\nInitial batch info:")
                    print(f"Images: shape={images.shape}, range=[{images.min():.3f}, {images.max():.3f}]")
                    print(f"Masks: shape={masks.shape}, range=[{masks.min():.3f}, {masks.max():.3f}]")
                    print(f"Mask unique values: {torch.unique(masks).tolist()}")
                    train_epoch.first_batch_done = True
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(images)
                
                # Calculate loss
                loss = criterion(outputs, masks)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"\nWarning: NaN loss detected in batch {batch_idx}")
                    continue
                
                # Backward pass
                loss.backward()
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                
                # Calculate metrics
                with torch.no_grad():
                    dice = dice_score(torch.sigmoid(outputs), masks)
                
                total_loss += loss.item()
                total_dice += dice
                
                # Update progress bar with current batch metrics
                current_loss = total_loss / (batch_idx + 1)
                current_dice = total_dice / (batch_idx + 1)
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'dice': f'{current_dice:.4f}'
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nWARNING: out of memory in batch {batch_idx}. Skipping batch")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    print(f"\nError in batch {batch_idx}: {str(e)}")
                    raise e
            
            except Exception as e:
                print(f"\nUnexpected error in batch {batch_idx}: {str(e)}")
                raise e
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return total_loss / (batch_idx + 1), total_dice / (batch_idx + 1)
    
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        raise e
    
    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches
    print(f"\nEpoch Summary:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average Dice: {avg_dice:.4f}")
    
    return avg_loss, avg_dice

def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_dice = 0
    num_batches = len(loader)
    
    try:
        print("\nValidating...")
        with torch.no_grad():
            pbar = tqdm(loader, desc='Validation', ncols=100)
            for batch_idx, (images, masks) in enumerate(pbar):
                try:
                    # Move to device
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    # Forward pass
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    
                    # Check for NaN loss
                    if torch.isnan(loss):
                        print("\nWarning: NaN loss detected during validation")
                        continue
                    
                    # Calculate metrics
                    dice = dice_score(torch.sigmoid(outputs), masks)
                    
                    total_loss += loss.item()
                    total_dice += dice
                    
                    # Update progress bar
                    current_loss = total_loss / (batch_idx + 1)
                    current_dice = total_dice / (batch_idx + 1)
                    pbar.set_postfix({
                        'val_loss': f'{current_loss:.4f}',
                        'val_dice': f'{current_dice:.4f}'
                    })
                
                except Exception as e:
                    print(f"\nError during validation: {str(e)}")
                    raise e
    
    except KeyboardInterrupt:
        print("\nValidation interrupted by user")
        return total_loss / (num_batches + 1), total_dice / (num_batches + 1)
    
    except Exception as e:
        print(f"\nValidation failed: {str(e)}")
        raise e
    
    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches
    print(f"\nValidation Summary:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average Dice: {avg_dice:.4f}")
    
    return avg_loss, avg_dice

def save_checkpoint(model, optimizer, epoch, loss, config, is_best=False):
    """
    Save model checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    if is_best:
        path = os.path.join(config.checkpoint_dir, 'best_model.pth')
    else:
        path = os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    
    torch.save(checkpoint, path)

def log_images(writer, images, masks, outputs, epoch, prefix='train'):
    """
    Log images to TensorBoard
    """
    # Convert outputs to binary masks
    pred_masks = (torch.sigmoid(outputs) > 0.5).float()
    
    # Add channel dimension for grayscale images if needed
    if images.dim() == 3:
        images = images.unsqueeze(1)
    if masks.dim() == 3:
        masks = masks.unsqueeze(1)
    if pred_masks.dim() == 3:
        pred_masks = pred_masks.unsqueeze(1)
    
    # Create a grid of images
    writer.add_images(f'{prefix}/images', images, epoch, dataformats='NCHW')
    writer.add_images(f'{prefix}/true_masks', masks, epoch, dataformats='NCHW')
    writer.add_images(f'{prefix}/pred_masks', pred_masks, epoch, dataformats='NCHW')

def main():
    config = TrainingConfig()
    
    # Set device - respect force_cpu setting
    if config.force_cpu:
        device = torch.device('cpu')
        print("Using CPU (forced)")
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using Apple Silicon GPU")
        else:
            device = torch.device('cpu')
            print("Using CPU")
    
    # Create transforms
    train_transform = create_transforms(is_train=True)
    val_transform = create_transforms(is_train=False)
    
    # Create datasets and dataloaders
    train_dataset = JSRTDataset(
        image_dir=os.path.join(config.train_data_dir, 'images'),
        mask_dir=os.path.join(config.train_data_dir, 'masks'),
        transform=train_transform
    )
    
    val_dataset = JSRTDataset(
        image_dir=os.path.join(config.val_data_dir, 'images'),
        mask_dir=os.path.join(config.val_data_dir, 'masks'),
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}\n')
    
    # Initialize model
    model = UNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        features=config.features,
        bilinear=config.bilinear,
        bn_momentum=config.bn_momentum,
        dropout_rate=config.dropout_rate
    ).to(device)
    
    # Initialize optimizer with weight decay
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2)
    )
    
    criterion = dice_loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=config.scheduler_patience,
        factor=config.scheduler_factor,
        min_lr=config.min_lr
    )
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(config.log_dir)
    
    # Training loop
    best_val_loss = float('inf')
    print("\nStarting training...")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Image size: {config.image_size}\n")
    
    for epoch in range(config.num_epochs):
        print(f'\nEpoch {epoch+1}/{config.num_epochs}')
        print('-' * 20)
        
        # Train
        train_loss, train_dice = train_epoch(
            model, train_loader, optimizer, criterion, device,
            gradient_clip=config.gradient_clip
        )
        
        # Validate
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Dice/train', train_dice, epoch)
        writer.add_scalar('Dice/val', val_dice, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        print(f"\nLearning rate: {current_lr:.2e}")
        
        # Log images periodically
        if epoch % config.log_images_every == 0:
            with torch.no_grad():
                # Log training images
                images, masks = next(iter(train_loader))
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                log_images(writer, images, masks, outputs, epoch, 'train')
                
                # Log validation images
                images, masks = next(iter(val_loader))
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                log_images(writer, images, masks, outputs, epoch, 'val')
        
        # Save checkpoints
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, config, is_best=True)
            print("\nSaved new best model!")
        
        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, config)
            print(f"\nSaved checkpoint at epoch {epoch+1}")
    
    writer.close()
    print('\nTraining completed!')

if __name__ == '__main__':
    main() 