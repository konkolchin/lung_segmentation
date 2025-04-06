import os
import shutil
import numpy as np
from PIL import Image
import requests
from tqdm import tqdm

def download_file(url, filename):
    """
    Download a file from a given URL showing progress
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

def setup_directories():
    """
    Create necessary directories for the dataset
    """
    directories = [
        'data/raw/jsrt',
        'data/raw/scr',
        'data/processed/images',
        'data/processed/masks',
        'data/train/images',
        'data/train/masks',
        'data/val/images',
        'data/val/masks'
    ]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def process_jsrt_image(image_path, output_path):
    """
    Process JSRT image from raw format to PNG
    JSRT images are 2048x2048 pixels, 12 bits per pixel, stored in big-endian format
    """
    try:
        # Read raw image data
        with open(image_path, 'rb') as f:
            raw_data = f.read()
        
        # Convert bytes to numpy array (16-bit unsigned integers, big-endian)
        image = np.frombuffer(raw_data, dtype='>u2')
        
        # Reshape to 2048x2048
        image = image.reshape(2048, 2048)
        
        # Normalize to 8-bit
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        # Save as PNG
        Image.fromarray(image).save(output_path)
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False

def process_scr_mask(mask_path, output_path):
    """
    Process SCR mask from TIF to PNG format
    """
    try:
        # Read TIF file
        mask = Image.open(mask_path)
        
        # Convert to binary
        mask_array = np.array(mask)
        mask_array = (mask_array > 127).astype(np.uint8) * 255
        
        # Save as PNG
        Image.fromarray(mask_array).save(output_path)
        return True
    except Exception as e:
        print(f"Error processing {mask_path}: {str(e)}")
        return False

def find_matching_masks(image_file, mask_files):
    """
    Find the corresponding lung field mask for an image
    """
    # Convert JPCLN to JPCNN to match mask naming
    mask_base = image_file.replace('JPCLN', 'JPCNN').replace('.IMG', '')
    # Find the corresponding lung field mask
    matching_masks = [m for m in mask_files if m.startswith(mask_base)]
    return matching_masks[0] if matching_masks else None

def split_dataset(images_dir, masks_dir, train_ratio=0.8, seed=42):
    """
    Split the dataset into training and validation sets
    Only split images that have corresponding masks
    """
    np.random.seed(seed)
    
    # Get all image files that have corresponding masks
    image_files = []
    for f in os.listdir(images_dir):
        if f.endswith('.png'):
            mask_path = os.path.join(masks_dir, f)
            if os.path.exists(mask_path):
                image_files.append(f)
    
    print(f"\nFound {len(image_files)} valid image-mask pairs")
    
    np.random.shuffle(image_files)
    
    # Split into train and validation
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # Move files to appropriate directories
    for file_list, subset in [(train_files, 'train'), (val_files, 'val')]:
        for img_file in file_list:
            # Move image
            src_img = os.path.join(images_dir, img_file)
            dst_img = os.path.join(f'data/{subset}/images', img_file)
            shutil.copy2(src_img, dst_img)
            
            # Move corresponding mask
            mask_file = img_file  # Masks will have the same name as images
            src_mask = os.path.join(masks_dir, mask_file)
            dst_mask = os.path.join(f'data/{subset}/masks', mask_file)
            shutil.copy2(src_mask, dst_mask)

def main():
    print("Setting up JSRT dataset...")
    
    # Create directory structure
    setup_directories()
    
    print("\nNote: Due to licensing requirements, please:")
    print("1. Download JSRT dataset manually from: http://db.jsrt.or.jp/eng.php")
    print("2. Download SCR masks from: https://zenodo.org/record/3902218")
    print("\nAfter downloading:")
    print("1. Place JSRT DICOM files in 'data/raw/jsrt/'")
    print("2. Place SCR mask files in 'data/raw/scr/'")
    print("3. Run this script again to process and organize the dataset")
    
    # Check if files exist
    if not os.path.exists('data/raw/jsrt') or not os.listdir('data/raw/jsrt'):
        print("\nNo JSRT images found in data/raw/jsrt/")
        return
    
    if not os.path.exists('data/raw/scr') or not os.listdir('data/raw/scr'):
        print("\nNo SCR masks found in data/raw/scr/")
        return
    
    # Process JSRT images
    print("\nProcessing JSRT images...")
    processed_images = 0
    jsrt_files = sorted([f for f in os.listdir('data/raw/jsrt') if f.endswith('.IMG')])
    
    for file in tqdm(jsrt_files):
        input_path = os.path.join('data/raw/jsrt', file)
        output_path = os.path.join('data/processed/images', file.replace('.IMG', '.png'))
        if process_jsrt_image(input_path, output_path):
            processed_images += 1
    
    print(f"\nProcessed {processed_images} JSRT images")
    
    # Process SCR masks
    print("\nProcessing SCR masks...")
    processed_masks = 0
    scr_files = sorted([f for f in os.listdir('data/raw/scr') if f.endswith('.tif')])
    
    # Process masks for each image
    for image_file in tqdm(jsrt_files, desc="Finding and processing matching masks"):
        mask_file = find_matching_masks(image_file, scr_files)
        if mask_file:
            input_path = os.path.join('data/raw/scr', mask_file)
            output_file = image_file.replace('.IMG', '.png')
            output_path = os.path.join('data/processed/masks', output_file)
            if process_scr_mask(input_path, output_path):
                processed_masks += 1
        else:
            print(f"Warning: No matching mask found for {image_file}")
    
    print(f"\nProcessed {processed_masks} SCR masks")
    print(f"Note: Each X-ray image should have one corresponding mask.")
    print(f"Total X-rays: {len(jsrt_files)}, Successfully processed masks: {processed_masks}")
    
    if processed_images > 0 and processed_masks > 0:
        # Split dataset
        print("\nSplitting dataset into train and validation sets...")
        split_dataset('data/processed/images', 'data/processed/masks')
        
        print("\nDataset setup complete!")
        print(f"Training images: {len(os.listdir('data/train/images'))}")
        print(f"Validation images: {len(os.listdir('data/val/images'))}")
    else:
        print("\nNo files were processed. Please check the input directories and file formats.")

if __name__ == "__main__":
    main() 