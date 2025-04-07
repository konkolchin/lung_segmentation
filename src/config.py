from dataclasses import dataclass
import os

@dataclass
class TrainingConfig:
    # Device settings
    force_cpu: bool = True  # Force CPU usage since GPU is too old
    
    # Model parameters
    in_channels: int = 1
    out_channels: int = 1
    features: list = None
    bilinear: bool = False  # Changed to use transposed convolutions
    
    # Training parameters
    batch_size: int = 2  # Reduced for CPU memory constraints
    num_epochs: int = 100
    learning_rate: float = 1e-3  # Increased learning rate
    weight_decay: float = 1e-5  # Reduced weight decay
    gradient_clip: float = 1.0
    
    # Dataset parameters
    train_data_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'prepared_data', 'train')
    val_data_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'prepared_data', 'val')
    image_size: tuple = (512, 512)  # Reduced image size to save memory
    
    # Augmentation parameters
    augmentation_prob: float = 0.7
    
    # Optimizer parameters
    beta1: float = 0.9
    beta2: float = 0.999
    
    # Learning rate scheduler parameters
    scheduler_patience: int = 3  # Reduced patience
    scheduler_factor: float = 0.5  # More aggressive reduction
    min_lr: float = 1e-6
    
    # Model parameters
    bn_momentum: float = 0.1
    dropout_rate: float = 0.1  # Reduced dropout for faster convergence
    
    # Checkpoint parameters
    checkpoint_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints')
    save_every: int = 5
    
    # Logging parameters
    log_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runs', 'lung_segmentation')
    log_images_every: int = 2
    
    def __post_init__(self):
        # Set default feature list if not provided
        if self.features is None:
            self.features = [48, 96, 192, 384]  # Increased feature maps for better capacity
        
        # Create necessary directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Minimal workers for CPU training
        self.num_workers = 0  # Set to 0 for CPU training to avoid memory issues
        self.pin_memory = False  # Disable pin_memory for CPU training 