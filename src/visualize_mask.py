import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

def visualize_mask(mask_path):
    # Load mask in different ways
    # 1. Direct grayscale
    mask1 = Image.open(mask_path).convert('L')
    arr1 = np.array(mask1)
    
    # 2. As RGB then convert
    mask2 = Image.open(mask_path).convert('RGB')
    arr2 = np.array(mask2)
    
    # 3. Direct without conversion
    mask3 = Image.open(mask_path)
    arr3 = np.array(mask3)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot each version
    axes[0].imshow(arr1, cmap='gray')
    axes[0].set_title(f'Grayscale\nShape: {arr1.shape}\nUnique: {np.unique(arr1)}')
    
    axes[1].imshow(arr2)
    axes[1].set_title(f'RGB\nShape: {arr2.shape}\nUnique: {np.unique(arr2)}')
    
    axes[2].imshow(arr3, cmap='gray')
    axes[2].set_title(f'Direct\nShape: {arr3.shape}\nUnique: {np.unique(arr3)}')
    
    plt.tight_layout()
    plt.savefig('mask_visualization.png')
    print(f"Saved visualization to mask_visualization.png")

if __name__ == "__main__":
    mask_path = "../data/train/masks/JPCLN001.png"
    visualize_mask(mask_path) 