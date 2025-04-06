# Running Lung Segmentation Training on Google Colab

This guide explains how to run the lung segmentation training pipeline on Google Colab.

## Data Preparation

Before uploading to Colab, organize your dataset using the provided script:

1. Prepare your source data:
   ```
   source_data/
   ├── images/
   │   ├── image1.png
   │   ├── image2.png
   │   └── ...
   └── masks/
       ├── image1.png
       ├── image2.png
       └── ...
   ```

2. Run the preparation script:
   ```bash
   python src/prepare_colab_data.py --source ./source_data --dest ./prepared_data --split 0.8
   ```
   This will:
   - Create train/val splits
   - Verify image-mask pairs
   - Organize files in the correct structure
   - Generate a summary report

3. Upload the prepared data:
   - Zip the prepared_data folder
   - Upload to Google Drive
   - The script will verify data integrity after copying to Colab

## Setup Instructions

1. Open the `lung_segmentation.ipynb` notebook in Google Colab
2. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. Clone the repository:
   ```bash
   !git clone https://github.com/YOUR_USERNAME/Lung_segmentation.git
   %cd Lung_segmentation
   ```

4. Install dependencies:
   ```bash
   !pip install -r requirements_colab.txt
   ```

5. Copy your dataset:
   ```bash
   # Create data directory
   !mkdir -p data
   
   # Unzip and copy the prepared dataset
   !unzip "/content/drive/MyDrive/prepared_data.zip" -d ./data
   
   # Verify the data structure
   !python src/prepare_colab_data.py --source ./data/prepared_data --dest ./data --split 0.8
   ```

## Directory Structure
The training script expects the following directory structure:
```
data/
├── train/
│   ├── images/
│   └── masks/
└── val/
    ├── images/
    └── masks/
```

## Training

The training script is configured to:
- Use Colab's GPU
- Save checkpoints to Google Drive
- Log training metrics and images to TensorBoard
- Use optimized batch size and workers for Colab

To start training:
```bash
cd src
python train.py
```

## Monitoring Training

1. Monitor training progress in real-time using TensorBoard:
```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/lung_segmentation_logs
```

2. Checkpoints are saved to:
   - `/content/drive/MyDrive/lung_segmentation_checkpoints`

## Optimizations for Colab

The code has been optimized for Colab with:
- Increased batch size (8)
- Multi-worker data loading
- Automatic GPU detection
- Persistent storage in Google Drive
- More frequent checkpointing
- Enhanced logging

## Troubleshooting

1. If you encounter OOM (Out of Memory) errors:
   - Reduce batch_size in config.py
   - Reduce num_workers if needed

2. If the training is interrupted:
   - Checkpoints are saved every 5 epochs
   - You can resume from the latest checkpoint

3. For any other issues:
   - Check the error messages in the notebook
   - Verify GPU availability with `!nvidia-smi`
   - Check your dataset paths 