from dataclasses import dataclass
import os

@dataclass
class TrainingConfig:
    # Device settings
    force_cpu: bool = True  # Force CPU usage
    
    # Model parameters
    in_channels: int = 1
    out_channels: int = 1
    features: list = None
    bilinear: bool = True
    
    # Training parameters
    batch_size: int = 4
    num_epochs: int = 100
    learning_rate: float = 5e-4  # Adjusted from 1e-3
    weight_decay: float = 1e-4  # Increased regularization
    gradient_clip: float = 1.0
    
    # Dataset parameters
    train_data_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'train')
    val_data_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'val')
    image_size: tuple = (512, 512)
    
    # Augmentation parameters
    augmentation_prob: float = 0.7  # Increased from 0.5
    
    # Optimizer parameters
    beta1: float = 0.9
    beta2: float = 0.999
    
    # Learning rate scheduler parameters
    scheduler_patience: int = 5  # Reduced from 8
    scheduler_factor: float = 0.3  # More aggressive LR reduction
    min_lr: float = 1e-6
    
    # Model parameters
    bn_momentum: float = 0.1
    dropout_rate: float = 0.2  # Added dropout
    
    # Checkpoint parameters
    checkpoint_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints')
    save_every: int = 10
    
    # Logging parameters
    log_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runs', 'lung_segmentation')
    log_images_every: int = 5
    
    def __post_init__(self):
        # Set default feature list if not provided
        if self.features is None:
            self.features = [64, 128, 256, 512]
        
        # Create necessary directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True) 