from dataclasses import dataclass
import os

@dataclass
class TrainingConfig:
    # Device settings
    force_cpu: bool = False  # Allow GPU usage by default
    
    # Model parameters
    in_channels: int = 1
    out_channels: int = 1
    features: list = None
    bilinear: bool = True
    
    # Training parameters
    batch_size: int = 8  # Increased for Colab GPU
    num_epochs: int = 100
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    
    # Dataset parameters
    train_data_dir: str = '/content/Lung_segmentation/data/train'  # Updated for Colab
    val_data_dir: str = '/content/Lung_segmentation/data/val'      # Updated for Colab
    image_size: tuple = (1024, 1024)
    
    # Augmentation parameters
    augmentation_prob: float = 0.7
    
    # Optimizer parameters
    beta1: float = 0.9
    beta2: float = 0.999
    
    # Learning rate scheduler parameters
    scheduler_patience: int = 5
    scheduler_factor: float = 0.3
    min_lr: float = 1e-6
    
    # Model parameters
    bn_momentum: float = 0.1
    dropout_rate: float = 0.2
    
    # Checkpoint parameters
    checkpoint_dir: str = '/content/drive/MyDrive/lung_segmentation_checkpoints'  # Save to Drive
    save_every: int = 5  # Save more frequently
    
    # Logging parameters
    log_dir: str = '/content/drive/MyDrive/lung_segmentation_logs'  # Save to Drive
    log_images_every: int = 2  # Log images more frequently
    
    def __post_init__(self):
        # Set default feature list if not provided
        if self.features is None:
            self.features = [64, 128, 256, 512]
        
        # Create necessary directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Enable multi-worker data loading for Colab
        self.num_workers = 2
        self.pin_memory = True 