import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use Agg backend
import matplotlib.pyplot as plt
from models.unet import UNet
from utils.dataset import JSRTDataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import os

def create_transforms():
    return A.Compose([
        A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})

def load_model(checkpoint_path):
    model = UNet(in_channels=1, out_channels=1)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def create_visualization(image, mask, prediction):
    # Convert tensors to numpy arrays
    image = (image.squeeze().numpy() * 0.229 + 0.485).clip(0, 1)
    mask = mask.squeeze().numpy()
    prediction = prediction.squeeze().numpy()
    
    # Create a figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.patch.set_facecolor('black')
    
    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original X-ray', color='white')
    axes[0, 0].axis('off')
    
    # Ground truth mask
    axes[0, 1].imshow(mask, cmap='hot')
    axes[0, 1].set_title('Ground Truth Mask', color='white')
    axes[0, 1].axis('off')
    
    # Model prediction
    axes[1, 0].imshow(prediction, cmap='hot')
    axes[1, 0].set_title('Model Prediction', color='white')
    axes[1, 0].axis('off')
    
    # Overlay prediction on original
    overlay = image.copy()
    overlay = np.stack([overlay] * 3, axis=-1)  # Convert to RGB
    overlay[..., 0] = np.maximum(overlay[..., 0], prediction * 0.7)  # Add red channel for prediction
    
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Overlay', color='white')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig

def main():
    # Paths
    base_dir = Path(__file__).parent.parent  # Get project root directory
    data_dir = base_dir / "prepared_data" / "val"  # Use prepared_data instead of data
    checkpoint_path = base_dir / "checkpoints" / "best_model.pth"
    output_dir = base_dir / "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Looking for data in: {data_dir}")
    print(f"Looking for checkpoint in: {checkpoint_path}")
    print(f"Output directory: {output_dir}")
    
    # Load model
    model = load_model(checkpoint_path)
    
    # Create dataset and dataloader
    dataset = JSRTDataset(
        image_dir=str(data_dir / "images"),
        mask_dir=str(data_dir / "masks"),
        transform=create_transforms()
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = str(output_dir / "lung_segmentation_demo.mp4")
    video = None
    
    print("Creating demonstration video...")
    for i, (image, mask) in enumerate(tqdm(dataloader)):
        # Get model prediction
        with torch.no_grad():
            prediction = torch.sigmoid(model(image))
        
        # Create visualization
        fig = create_visualization(image, mask, prediction)
        
        # Convert matplotlib figure to image
        fig.canvas.draw()
        # Get the RGBA buffer from the figure
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        # Reshape it to a Height x Width x 4 array
        buf = buf.reshape(h, w, 4)
        # Convert RGBA to BGR
        frame = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        
        # Initialize video writer with first frame
        if video is None:
            height, width = frame.shape[:2]
            video = cv2.VideoWriter(video_path, fourcc, 2.0, (width, height))
        
        # Write frame multiple times to slow down video
        for _ in range(15):  # Each prediction shown for ~0.5 seconds
            video.write(frame)
        
        plt.close()
    
    video.release()
    print(f"Video saved to {video_path}")
    print("Note: You may need to convert the video to a web-compatible format using:")
    print(f"ffmpeg -i {video_path} -vcodec libx264 {video_path.replace('.mp4', '_web.mp4')}")

if __name__ == "__main__":
    main() 