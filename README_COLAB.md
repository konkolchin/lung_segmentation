# Running Lung Segmentation Training on Google Colab

This guide explains how to run the lung segmentation training on Google Colab with GPU acceleration.

## Setup Steps

1. **Upload Dataset to Google Drive**
   - Create a folder in your Google Drive called `lung_segmentation_data`
   - Inside it, create the following structure:
     ```
     lung_segmentation_data/
     ├── data/
     │   ├── train/
     │   │   ├── images/
     │   │   └── masks/
     │   └── val/
     │       ├── images/
     │       └── masks/
     ```
   - Upload your training and validation data to the respective folders

2. **Open the Notebook in Colab**
   - Open Google Colab (https://colab.research.google.com)
   - Upload `lung_segmentation_training.ipynb` or open it from your GitHub repository
   - Make sure you're using a GPU runtime:
     - Runtime → Change runtime type → Hardware accelerator → GPU

3. **Mount Google Drive**
   - The notebook will automatically mount your Google Drive
   - When prompted, authorize access to your Drive

4. **Run the Training**
   - Run all cells in sequence
   - The training progress will be displayed in the output
   - You can monitor training metrics using TensorBoard

## Features

- Automatic GPU detection and utilization
- Increased batch size for GPU training
- TensorBoard integration for monitoring:
  - Training and validation loss
  - Dice scores
  - Learning rate
  - Sample predictions
- Checkpointing of best models
- Google Drive integration for data storage

## Troubleshooting

1. **GPU Not Available**
   - Verify you've selected GPU runtime in Colab
   - Check if you've exceeded your Colab GPU quota
   - Try reconnecting to the runtime

2. **Out of Memory**
   - Reduce batch size in `ColabConfig`
   - Clear output cells and restart runtime
   - Try using a smaller image size

3. **Data Loading Issues**
   - Verify your Google Drive folder structure
   - Check file permissions
   - Ensure all image files are in the correct format

## Tips

- Save important checkpoints to your Google Drive
- Use TensorBoard to monitor training progress
- For long training sessions, enable "Settings → Hardware accelerator → GPU" to prevent timeout
- Consider using Colab Pro for longer sessions and better GPUs 