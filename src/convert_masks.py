import os
import shutil
import numpy as np
from PIL import Image
import urllib.request
import zipfile
from tqdm import tqdm
import requests

def download_masks():
    """Download masks.zip from Zenodo"""
    url = "https://zenodo.org/records/7056076/files/masks.zip"
    print(f"Downloading masks from {url}...")
    
    temp_zip = "temp_masks.zip"
    
    # Use requests for better download handling
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress bar
        with open(temp_zip, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        
        # Verify the file exists and has content
        if not os.path.exists(temp_zip) or os.path.getsize(temp_zip) == 0:
            raise Exception("Download failed - empty or missing file")
            
        return temp_zip
        
    except Exception as e:
        print(f"Error downloading file: {e}")
        if os.path.exists(temp_zip):
            os.remove(temp_zip)
        raise

def verify_mask(mask_path):
    """Verify that a mask file is valid"""
    try:
        mask = Image.open(mask_path)
        mask_array = np.array(mask)
        print(f"\nVerifying mask: {os.path.basename(mask_path)}")
        print(f"Shape: {mask_array.shape}")
        print(f"dtype: {mask_array.dtype}")
        print(f"Unique values: {np.unique(mask_array)}")
        print(f"Min: {mask_array.min()}, Max: {mask_array.max()}")
        return True
    except Exception as e:
        print(f"Error verifying mask {mask_path}: {e}")
        return False

def extract_and_convert_masks(zip_path, output_dir):
    """Extract and convert masks to proper format"""
    # Create temporary extraction directory
    temp_dir = os.path.abspath("temp_masks")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    # Create output directory
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    try:
        # Extract zip file
        print(f"Extracting {zip_path} to {temp_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List contents before extracting
            contents = zip_ref.namelist()
            print(f"Zip contains {len(contents)} files")
            print("First few files:", contents[:5])
            
            zip_ref.extractall(temp_dir)
        
        # Process each mask file
        mask_files = []
        for root, dirs, files in os.walk(temp_dir):
            for f in files:
                if f.endswith('.tif'):  # Changed from .png to .tif
                    mask_files.append(os.path.join(root, f))
        
        print(f"Found {len(mask_files)} mask files")
        
        if not mask_files:
            raise Exception("No mask files found after extraction!")
        
        for mask_path in tqdm(mask_files, desc="Converting masks"):
            try:
                # Load TIF mask
                mask = Image.open(mask_path)
                mask_array = np.array(mask)
                
                # Print info for first file
                if mask_path == mask_files[0]:
                    print(f"\nFirst mask info:")
                    print(f"Original shape: {mask_array.shape}")
                    print(f"Original dtype: {mask_array.dtype}")
                    print(f"Original range: [{mask_array.min()}, {mask_array.max()}]")
                    print(f"Original unique values: {np.unique(mask_array)}")
                    print(f"Sample 5x5 corner:")
                    print(mask_array[:5, :5])
                
                # Convert to binary (0 or 255 for PNG)
                binary_mask = (mask_array > 0).astype(np.uint8) * 255
                
                # Create output path
                mask_file = os.path.basename(mask_path)
                output_path = os.path.join(output_dir, mask_file.replace('.tif', '.png'))
                
                # Save as PNG
                output_img = Image.fromarray(binary_mask, mode='L')
                output_img.save(output_path, format='PNG')
                
                # Verify the saved file
                if mask_path == mask_files[0]:
                    # Verify the first file was saved correctly
                    saved_mask = Image.open(output_path)
                    saved_array = np.array(saved_mask)
                    print(f"\nSaved mask info:")
                    print(f"Shape: {saved_array.shape}")
                    print(f"dtype: {saved_array.dtype}")
                    print(f"range: [{saved_array.min()}, {saved_array.max()}]")
                    print(f"unique values: {np.unique(saved_array)}")
                    print(f"Sample 5x5 corner:")
                    print(saved_array[:5, :5])
                    print(f"Saved to: {output_path}")
                
            except Exception as e:
                print(f"Error processing {mask_path}: {e}")
        
    except Exception as e:
        print(f"Error during extraction/conversion: {e}")
        raise
    finally:
        # Clean up
        print("\nCleaning up temporary files...")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    print(f"\nMask conversion completed. Converted masks are in {output_dir}")
    
    # Verify some converted files
    print("\nVerifying converted files...")
    converted_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    print(f"Number of converted files: {len(converted_files)}")
    if converted_files:
        print("Checking first few files:")
        for f in converted_files[:3]:
            verify_mask(os.path.join(output_dir, f))

if __name__ == "__main__":
    # Define directories using absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    train_masks_dir = os.path.join(project_root, "data", "train", "masks")
    val_masks_dir = os.path.join(project_root, "data", "val", "masks")
    
    # Create directories if they don't exist
    os.makedirs(train_masks_dir, exist_ok=True)
    os.makedirs(val_masks_dir, exist_ok=True)
    
    print(f"Train masks directory: {train_masks_dir}")
    print(f"Val masks directory: {val_masks_dir}")
    
    try:
        # Download and process masks
        zip_path = download_masks()
        
        # Convert masks for both train and val directories
        print("\nConverting training masks...")
        extract_and_convert_masks(zip_path, train_masks_dir)
        
        print("\nConverting validation masks...")
        extract_and_convert_masks(zip_path, val_masks_dir)
        
    except Exception as e:
        print(f"\nError during conversion process: {e}")
    finally:
        # Clean up zip file
        if os.path.exists("temp_masks.zip"):
            os.remove("temp_masks.zip") 