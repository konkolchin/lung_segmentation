import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_image_mask_pair(image_path, mask_path, target_size=(1024, 1024)):
    """
    Load an image and its corresponding mask, resizing both to the same dimensions
    """
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    
    # Resize both to the same dimensions
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    mask = mask.resize(target_size, Image.Resampling.NEAREST)
    
    return np.array(image), np.array(mask)

def visualize_pairs(data_dir, num_samples=3):
    """
    Visualize random image-mask pairs from the dataset
    """
    # Get list of image files
    image_dir = os.path.join(data_dir, 'images')
    mask_dir = os.path.join(data_dir, 'masks')
    
    image_files = os.listdir(image_dir)
    
    # Randomly select samples
    samples = random.sample(image_files, min(num_samples, len(image_files)))
    
    # Create a figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle('Image-Mask Pair Visualization', fontsize=16)
    
    for idx, image_file in enumerate(samples):
        # Load image and mask
        image_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, image_file)
        
        image, mask = load_image_mask_pair(image_path, mask_path)
        
        # Create overlay
        overlay = np.zeros_like(image)
        overlay[mask > 0] = 255
        
        # Plot original image
        axes[idx, 0].imshow(image, cmap='gray')
        axes[idx, 0].set_title(f'Original X-ray\n{image_file}')
        axes[idx, 0].axis('off')
        
        # Plot mask
        axes[idx, 1].imshow(mask, cmap='gray')
        axes[idx, 1].set_title('Segmentation Mask')
        axes[idx, 1].axis('off')
        
        # Plot overlay
        axes[idx, 2].imshow(image, cmap='gray')
        axes[idx, 2].imshow(overlay, cmap='Reds', alpha=0.3)
        axes[idx, 2].set_title('Overlay')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    output_dir = 'visualizations'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{os.path.basename(data_dir)}_visualization.png'), 
                bbox_inches='tight', dpi=300)
    print(f"Visualization saved to {output_dir}/{os.path.basename(data_dir)}_visualization.png")
    plt.close()

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Visualize samples from both training and validation sets
    print("Visualizing training samples...")
    visualize_pairs('data/train', num_samples=3)
    
    print("\nVisualizing validation samples...")
    visualize_pairs('data/val', num_samples=3) 