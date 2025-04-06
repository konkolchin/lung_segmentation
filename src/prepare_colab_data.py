import os
import shutil
import argparse
from pathlib import Path
import sys

def create_directory_structure(base_path):
    """Create the required directory structure."""
    dirs = [
        'train/images',
        'train/masks',
        'val/images',
        'val/masks'
    ]
    
    for dir_path in dirs:
        full_path = os.path.join(base_path, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"Created directory: {full_path}")

def verify_image_mask_pairs(image_dir, mask_dir):
    """Verify that all images have corresponding masks."""
    images = set(os.listdir(image_dir))
    masks = set(os.listdir(mask_dir))
    
    missing_masks = [img for img in images if img not in masks]
    missing_images = [mask for mask in masks if mask not in images]
    
    if missing_masks:
        print(f"\nWarning: Found {len(missing_masks)} images without corresponding masks:")
        for img in missing_masks[:5]:
            print(f"  - {img}")
        if len(missing_masks) > 5:
            print(f"  ... and {len(missing_masks) - 5} more")
    
    if missing_images:
        print(f"\nWarning: Found {len(missing_images)} masks without corresponding images:")
        for mask in missing_images[:5]:
            print(f"  - {mask}")
        if len(missing_images) > 5:
            print(f"  ... and {len(missing_images) - 5} more")
    
    return len(missing_masks) == 0 and len(missing_images) == 0

def organize_data(source_dir, dest_dir, split_ratio=0.8):
    """Organize data into train and validation sets."""
    # Create directory structure
    create_directory_structure(dest_dir)
    
    # Get all image files
    image_files = [f for f in os.listdir(os.path.join(source_dir, 'images'))
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Calculate split
    num_train = int(len(image_files) * split_ratio)
    train_files = image_files[:num_train]
    val_files = image_files[num_train:]
    
    # Copy files
    for file_list, subset in [(train_files, 'train'), (val_files, 'val')]:
        for img_file in file_list:
            # Copy image
            src_img = os.path.join(source_dir, 'images', img_file)
            dst_img = os.path.join(dest_dir, subset, 'images', img_file)
            shutil.copy2(src_img, dst_img)
            
            # Copy mask
            src_mask = os.path.join(source_dir, 'masks', img_file)
            dst_mask = os.path.join(dest_dir, subset, 'masks', img_file)
            if os.path.exists(src_mask):
                shutil.copy2(src_mask, dst_mask)
    
    # Verify data
    print("\nVerifying train set...")
    train_ok = verify_image_mask_pairs(
        os.path.join(dest_dir, 'train', 'images'),
        os.path.join(dest_dir, 'train', 'masks')
    )
    
    print("\nVerifying validation set...")
    val_ok = verify_image_mask_pairs(
        os.path.join(dest_dir, 'val', 'images'),
        os.path.join(dest_dir, 'val', 'masks')
    )
    
    # Print summary
    print("\nData Organization Summary:")
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    print(f"Total samples: {len(image_files)}")
    print(f"Split ratio: {split_ratio:.2f}")
    print(f"Data verification: {'✓ Passed' if train_ok and val_ok else '✗ Failed'}")

def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for Colab training')
    parser.add_argument('--source', type=str, required=True,
                      help='Source directory containing images and masks subdirectories')
    parser.add_argument('--dest', type=str, required=True,
                      help='Destination directory for organized dataset')
    parser.add_argument('--split', type=float, default=0.8,
                      help='Train/validation split ratio (default: 0.8)')
    
    args = parser.parse_args()
    
    # Verify source directory structure
    if not os.path.exists(args.source):
        print(f"Error: Source directory '{args.source}' does not exist")
        sys.exit(1)
    
    if not all(os.path.exists(os.path.join(args.source, d)) for d in ['images', 'masks']):
        print("Error: Source directory must contain 'images' and 'masks' subdirectories")
        sys.exit(1)
    
    # Organize data
    print(f"\nOrganizing data from {args.source} to {args.dest}")
    print(f"Using train/val split ratio of {args.split}")
    organize_data(args.source, args.dest, args.split)

if __name__ == '__main__':
    main() 