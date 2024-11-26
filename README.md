# Action Recognition Demo

This repository provides a demo for action recognition using the [OpenMMLab](https://openmmlab.com/) framework, specifically leveraging [MMAction2](https://github.com/open-mmlab/mmaction2). The main script performs action recognition on an input video and generates an output video with the predicted label overlaid.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Directory Structure](#directory-structure)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Overview

The main functionalities of this project include:

- **Action Recognition**: Classify actions in videos using pre-trained models.
- **Visualization**: Overlay predicted labels onto the input video.
- **Customization**: Adjust font size, color, resolution, and more.

## Features

- Easy setup with clear instructions.
- Supports CPU and GPU inference.
- Customizable output video generation.
- Uses MMAction2 and other OpenMMLab tools for robust performance.

## Prerequisites

- **Operating System**: Linux or macOS (Windows is not officially supported by MMAction2)
- **Python**: 3.8 or higher
- **Git**: For cloning repositories
- **Conda**: Anaconda or Miniconda for environment management
- **CUDA Toolkit**: Required if you plan to run inference on a GPU (skip if using CPU)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/actionRecognition2025.git
cd actionRecognition2025
```