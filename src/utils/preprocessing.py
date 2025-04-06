import cv2
import numpy as np
from skimage import exposure
from PIL import Image

def preprocess_xray(image_path, target_size=(512, 512)):
    """
    Preprocess X-ray image for lung segmentation
    
    Args:
        image_path (str): Path to the X-ray image
        target_size (tuple): Target size for resizing
    
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Read image
    if isinstance(image_path, str):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = image_path
        
    if image is None:
        raise ValueError("Could not read the image")

    # Resize image
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = clahe.apply(image)
    
    # Normalize pixel values
    image = image / 255.0
    
    # Add channel dimension
    image = np.expand_dims(image, axis=0)
    
    return image.astype(np.float32)

def enhance_bronchi(image):
    """
    Enhance bronchi structures in the X-ray image
    
    Args:
        image (numpy.ndarray): Input image
    
    Returns:
        numpy.ndarray: Enhanced image
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Apply adaptive histogram equalization
    enhanced = exposure.equalize_adapthist(blurred)
    
    # Apply unsharp masking for edge enhancement
    gaussian_3 = cv2.GaussianBlur(enhanced, (9, 9), 2.0)
    unsharp_image = cv2.addWeighted(enhanced, 1.5, gaussian_3, -0.5, 0)
    
    return unsharp_image

def post_process_mask(mask, min_size=100):
    """
    Post-process the predicted mask
    
    Args:
        mask (numpy.ndarray): Predicted binary mask
        min_size (int): Minimum size of connected components to keep
    
    Returns:
        numpy.ndarray: Processed binary mask
    """
    # Convert to binary
    mask = (mask > 0.5).astype(np.uint8)
    
    # Remove small connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_size:
            mask[labels == i] = 0
            
    # Apply morphological operations
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPHOPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPHCLOSE, kernel)
    
    return mask 