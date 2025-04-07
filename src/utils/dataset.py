import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class JSRTDataset(Dataset):
    """
    Dataset class for JSRT chest X-ray segmentation dataset
    """
    def __init__(self, image_dir, mask_dir, transform=None, target_size=(1024, 1024)):
        """
        Args:
            image_dir (str): Path to the image directory
            mask_dir (str): Path to the mask directory
            transform (callable, optional): Optional transform to be applied on a sample
            target_size (tuple): Target size for resizing images and masks
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = target_size
        
        # Get all valid image-mask pairs
        self.valid_files = []
        for img_name in os.listdir(image_dir):
            mask_path = os.path.join(mask_dir, img_name)  # Use same filename for mask
            img_path = os.path.join(image_dir, img_name)
            
            if os.path.exists(mask_path):
                try:
                    # Try to open both files to verify they're valid
                    img = Image.open(img_path)
                    mask = Image.open(mask_path)
                    img.close()
                    mask.close()
                    self.valid_files.append(img_name)
                except Exception as e:
                    print(f"Warning: Skipping {img_name} due to error: {e}")
        
        print(f"Found {len(self.valid_files)} valid image-mask pairs")
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (image, mask) where mask is the segmentation mask
        """
        img_name = self.valid_files[idx]
        
        # Load image and mask (using same filename)
        image_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # Debug info for first item
        if idx == 0:
            print(f"\nLoading files:")
            print(f"Image: {image_path}")
            print(f"Mask: {mask_path}")
            print(f"Mask exists: {os.path.exists(mask_path)}")
        
        # Load image and mask
        image = Image.open(image_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        
        # Resize both image and mask to target size
        image = image.resize(self.target_size, Image.Resampling.BILINEAR)
        mask = mask.resize(self.target_size, Image.Resampling.NEAREST)
        
        # Convert to numpy arrays
        image = np.array(image)
        mask = np.array(mask)
        
        # Debug info for first item
        if idx == 0:
            print(f"\nArray info:")
            print(f"Image: shape={image.shape}, dtype={image.dtype}, range=[{image.min()}, {image.max()}]")
            print(f"Mask: shape={mask.shape}, dtype={mask.dtype}, range=[{mask.min()}, {mask.max()}]")
            print(f"Mask unique values: {np.unique(mask)}")
        
        # Normalize mask to [0, 1]
        mask = (mask > 127).astype(np.float32)
        
        # Apply transforms
        if self.transform:
            # Convert image to float32 for transforms
            image = image.astype(np.float32)
            
            # Apply transforms
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            
            # Add channel dimension to mask if needed
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            
            # Debug info for first item
            if idx == 0:
                print(f"\nTransformed tensors:")
                print(f"Image: shape={image.shape}, range=[{image.min():.3f}, {image.max():.3f}]")
                print(f"Mask: shape={mask.shape}, range=[{mask.min():.3f}, {mask.max():.3f}]")
                print(f"Mask unique values: {torch.unique(mask).tolist()}")
        else:
            # Convert to tensors manually
            image = torch.from_numpy(image).float().unsqueeze(0) / 255.0
            mask = torch.from_numpy(mask).float().unsqueeze(0)
        
        return image, mask 