import numpy as np
import os

def check_raw_file(file_path):
    # Read the file in binary mode
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    
    print(f"\nChecking file: {os.path.basename(file_path)}")
    print(f"File size: {len(raw_data)} bytes")
    
    # Try to interpret as different numpy dtypes
    print("\nTrying different interpretations:")
    
    # Try uint8
    arr_uint8 = np.frombuffer(raw_data, dtype=np.uint8)
    print(f"\nAs uint8:")
    print(f"Shape: {arr_uint8.shape}")
    print(f"Unique values: {np.unique(arr_uint8)[:10]}...")
    print(f"Min: {arr_uint8.min()}, Max: {arr_uint8.max()}")
    
    # Try uint16
    if len(raw_data) >= 2:
        arr_uint16 = np.frombuffer(raw_data, dtype=np.uint16)
        print(f"\nAs uint16:")
        print(f"Shape: {arr_uint16.shape}")
        print(f"Unique values: {np.unique(arr_uint16)[:10]}...")
        print(f"Min: {arr_uint16.min()}, Max: {arr_uint16.max()}")
    
    # Try float32
    if len(raw_data) >= 4:
        arr_float32 = np.frombuffer(raw_data, dtype=np.float32)
        print(f"\nAs float32:")
        print(f"Shape: {arr_float32.shape}")
        print(f"Unique values: {np.unique(arr_float32)[:10]}...")
        print(f"Min: {arr_float32.min()}, Max: {arr_float32.max()}")
    
    # Check if it might be a 1024x1024 image
    if len(raw_data) == 1024*1024:
        print("\nLooks like it could be a 1024x1024 uint8 image!")
        arr_img = arr_uint8.reshape(1024, 1024)
        print(f"As 1024x1024:")
        print(f"Shape: {arr_img.shape}")
        print(f"Unique values: {np.unique(arr_img)[:10]}...")
        print(f"Min: {arr_img.min()}, Max: {arr_img.max()}")
    
    # Print first few bytes for inspection
    print("\nFirst 32 bytes:")
    print(" ".join(f"{b:02x}" for b in raw_data[:32]))

if __name__ == "__main__":
    mask_path = "../data/train/masks/JPCLN001.png"
    check_raw_file(mask_path) 