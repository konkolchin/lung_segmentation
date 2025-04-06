# Lung Segmentation Project

A deep learning project for automatic lung segmentation in chest X-ray images using PyTorch and U-Net architecture.

## Features

- PyTorch-based U-Net implementation
- JSRT dataset support
- Real-time training monitoring with TensorBoard
- Automatic GPU/CPU detection
- Docker support for easy deployment
- Comprehensive data augmentation pipeline
- Dice loss and score metrics
- Checkpoint saving and loading
- Learning rate scheduling

## Project Structure

```
lung_segmentation/
├── src/
│   ├── models/
│   │   └── unet.py
│   ├── utils/
│   │   └── dataset.py
│   ├── config.py
│   └── train.py
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── val/
│       ├── images/
│       └── masks/
├── checkpoints/
├── logs/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Installation

### Option 1: Local Installation

1. Clone the repository:
```bash
git clone https://github.com/konkolchin/lung_segmentation.git
cd lung_segmentation
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Option 2: Docker Installation

1. Install Docker and NVIDIA Container Toolkit (for GPU support)

2. Build and run using docker-compose:
```bash
docker-compose build
docker-compose up
```

## Dataset Preparation

1. Download the JSRT dataset
2. Run the conversion script:
```bash
python src/convert_masks.py
```

This will:
- Download the mask files
- Convert them to the correct format
- Split them into training and validation sets

## Training

### Local Training

```bash
python src/train.py
```

### Docker Training

```bash
docker-compose run lung_segmentation
```

## Configuration

Key parameters can be modified in `src/config.py`:

- Learning rate
- Batch size
- Number of epochs
- Model architecture
- Data augmentation
- Device settings

## Monitoring

Training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir logs
```

This will show:
- Training/validation loss
- Dice scores
- Learning rate
- Sample predictions

## Model Architecture

The project uses a U-Net architecture with:
- Configurable input/output channels
- Batch normalization
- Optional bilinear upsampling
- Dropout for regularization

## Results

The model achieves:
- Dice score on validation set
- Training/validation loss curves
- Sample segmentation results

## Docker Support

The project includes Docker support for reproducible environments:
- CUDA-enabled PyTorch base image
- Automatic GPU detection
- Volume mounting for data and checkpoints
- Easy configuration through environment variables

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- JSRT dataset providers
- PyTorch team
- TensorBoard developers 