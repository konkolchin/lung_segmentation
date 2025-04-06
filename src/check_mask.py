from PIL import Image
import numpy as np
import os

def check_mask(mask_path):
    print(f"\nChecking mask: {mask_path}")
    print(f"File exists: {os.path.exists(mask_path)}")
    
    if not os.path.exists(mask_path):
        return
    
    print(f"File size: {os.path.getsize(mask_path)} bytes")
    
    # Load mask
    mask = Image.open(mask_path)
    print(f"Original mode: {mask.mode}")
    print(f"Original size: {mask.size}")
    
    # Convert to grayscale
    mask_gray = mask.convert('L')
    mask_array = np.array(mask_gray)
    
    print(f"\nMask array info:")
    print(f"Shape: {mask_array.shape}")
    print(f"dtype: {mask_array.dtype}")
    print(f"Unique values: {np.unique(mask_array)}")
    print(f"Min: {mask_array.min()}")
    print(f"Max: {mask_array.max()}")
    
    print("\nSample 5x5 corner:")
    print(mask_array[:5, :5])
    
    # Convert to binary
    binary_mask = (mask_array > 127).astype(np.float32)
    print("\nAfter binarization:")
    print(f"Unique values: {np.unique(binary_mask)}")
    print("Sample 5x5 corner:")
    print(binary_mask[:5, :5])

if __name__ == "__main__":
    # Check a mask file
    mask_path = "../data/train/masks/JPCLN001.png"
    check_mask(mask_path) 