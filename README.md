# TrapSalNet: A Lightweight Transformer-Driven Multi-level Trapezoidal Attention Network for Efficient Saliency Detection


## Introduction

Salient Object Detection (SOD) has seen significant advancements with the development of deep convolutional neural networks (CNNs) and transformer-based techniques. However, state-of-the-art (SOTA) methods often struggle to balance computational efficiency with high performance, particularly in capturing complex features and long-range dependencies. **SalNet** addresses these challenges through a lightweight pyramid vision transformer backbone, alongside several novel components for enhanced feature extraction and fusion.

## Key Features

- **Lightweight Pyramid Vision Transformer Backbone**: Efficiently extracts multi-scale features while maintaining a balance between computational cost and performance.
- **Contextual Feature Refinement Block (CFRB)**: Utilizes dilated convolutions to capture rich contextual information at each scale.
- **Coordinate Attention (COA) Module**: Highlights important spatial locations in the initial feature maps to improve spatial feature learning.
- **Efficient Multi-Headed Self-Attention (EMHSA) Module**: Captures long-range dependencies and global context efficiently.
- **Efficient Channel Attention (ECA) Module**: Reduces information loss during feature fusion by adaptively recalibrating channel-wise feature responses.
- **High-Resolution Saliency Map Generation**: Features are progressively upsampled and concatenated to produce a detailed saliency map.

## Benchmark Performance

SalNet outperforms over 30 SOTA SOD methods across six benchmark RGB datasets. It achieves superior performance while maintaining a lightweight architecture, making it suitable for deployment on resource-constrained platforms.

## Installation

To clone and set up this project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/TalhaUsman-ZERO/SalNet.git
    ```

2. Navigate to the project directory:
    ```bash
    cd SalNet
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Once you have installed the dependencies, you can run the following commands to train or test the model:

- **To train the model:**
    ```bash
    python train.py --config config/train_config.yaml
    ```

- **To test the model:**
    ```bash
    python test.py --config config/test_config.yaml
    ```

Refer to the `config` directory for configuration files that can be customized for different datasets and settings.

## Results and Pre-trained Models

Pre-trained models and results on benchmark datasets can be found in the [releases section](https://github.com/TalhaUsman-ZERO/SalNet/releases) of this repository.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{Usman2024SalNet,
  title={TrapSalNet: A Lightweight Transformer-Driven Multi-level Trapezoidal Attention Network for Efficient Saliency Detection},
  author={Usman, Talha and others},
  journal={Journal Name},
  year={2024}
}
