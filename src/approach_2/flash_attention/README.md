# Flash Attention Vision Transformer

This directory contains a variant of the Approach 2 Vision Transformer model optimized for GPUs with high compute capability

## Architecture Changes

The core architecture remains a **Vision Transformer (ViT) Encoder** paired with a **Transformer Decoder**. The key difference is in the `MultiHeadAttention` layers:

-   **Flash Attention**: We explicitly set `use_flash_attention=True` in the `MultiHeadAttention` layers.
    -   **Mechanism**: This implementation uses a memory-efficient attention algorithm that reduces HBM (High Bandwidth Memory) accesses, leading to faster training and lower memory usage.
    -   **Fallback**: The code includes a robust fallback mechanism. If your TensorFlow/Keras version or hardware doesn't support the explicit argument, it gracefully reverts to standard attention without crashing.
-   **XLA Compilation**: The training loop (`train.py`) enables XLA (Accelerated Linear Algebra) compilation via `jit_compile=True`. This fuses operations and optimizes kernel execution, providing significant speedups on V100 GPUs.

## Running on NVIDIA GPU

### 1. Environment Setup
Ensure you have a compatible Python environment with TensorFlow and Keras installed.
```bash
# Example setup (adjust based on your HPC modules)
module load python/3.9 cuda/11.8 cudnn/8.6
pip install tensorflow keras --upgrade
```

### 2. Run Training
Execute the training script from the root of the project (so python can resolve `src` imports):

```bash
# Run from /path/to/iml-a3/
python3 src/approach_2/flash_attention/train.py
```
