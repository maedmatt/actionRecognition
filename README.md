# Action Recognition Video Classifier

## Overview

This Python script uses MMAction2 to perform action recognition on video files. It can classify human actions in videos using a pre-trained model from the Kinetics-400 dataset.

## Features

- Action classification using a Temporal Segment Network (TSN) model
- Support for processing local video files
- Configurable output with custom visualization options
- Generates annotated video or GIF with predicted action label

## Prerequisites

### System Requirements
- Python 3.7+
- CUDA (optional, for GPU acceleration)

### Dependencies

- PyTorch
- MMEngine
- MMAction2
- moviepy
- OpenCV

## Installation

### 1. Create a Virtual Environment (Recommended)

```bash
# Create a new Conda environment
conda create -n action_recognition_env python=3.8 -y

# Activate the environment
conda activate action_recognition_env
```

```bash
git clone https://github.com/your-username/actionRecognition2025.git
cd actionRecognition2025
```
## 2. Install Dependencies

```bash
# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio

# Install MMEngine and MMAction2
pip install mmengine
pip install -U openmim
mim install mmcv
mim install mmaction2

# Additional dependencies
pip install moviepy opencv-python-headless
```
## 3. Download Pre-trained Model and Config Files

The script requires:

- **Model configuration file**
- **Checkpoint file**
- **Label map**

You can download these from the MMAction2 model zoo:

- **Config**: [configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py](https://github.com/open-mmlab/mmaction2/tree/main/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py)
- **Checkpoint**: [tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth)
- **Label Map**: [Kinetics-400 action labels](https://github.com/open-mmlab/mmaction2/blob/main/tools/data/kinetics/label_map.txt)

### 4. Run the Script

#### Basic Usage

```bash
# Ensure you are in the project directory and virtual environment is activated
python action_recognition_script.py
```

#### Configuration Examples

1. **Basic Configuration**
```python
# Minimal configuration in main() function
args.video = 'input/example_video.mp4'  # Path to your input video
args.device = 'cpu'  # Use CPU (change to 'cuda:0' for GPU)
```

2. **Advanced Configuration**
```python
# More detailed configuration options
args.video = 'input/sports_action.mp4'
args.device = 'cuda:0'  # Use GPU
args.fps = 25  # Custom frame rate
args.font_color = 'red'  # Change annotation color
args.target_resolution = (720, -1)  # Resize to 720p width, maintain aspect ratio
```

#### Common Run Scenarios

1. **Process Local Video**
```bash
python action_recognition_script.py
```

2. **GPU Processing (if CUDA available)**
```bash
CUDA_VISIBLE_DEVICES=0 python action_recognition_script.py
```

#### Expected Output

When successful, you'll see:
```
The top label with corresponding score is:
[Action Label]: [Confidence Score]

# Output video will be generated in the 'output/' directory
```

#### Potential Errors and Solutions

1. **FileNotFoundError**
   - Ensure all paths (video, config, checkpoint) are correct
   - Use absolute paths if relative paths fail

2. **CUDA Errors**
   - Verify CUDA installation
   - Check PyTorch and CUDA version compatibility
   - Fallback to CPU mode if GPU processing fails

3. **Dependency Issues**
   - Reinstall dependencies
   - Verify versions of PyTorch, MMAction2, and other libraries

## Acknowledgements

- **[MMAction2 Project](https://github.com/open-mmlab/mmaction2)**  
- **[Kinetics-400 Dataset](https://deepmind.com/research/open-source/kinetics)**
