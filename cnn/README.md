# CNN from Scratch

A convolutional neural network with **manually implemented forward and backward passes** for Conv2d, ReLU, and Batch Normalization. Trained on MNIST digit classification.

## What's implemented from scratch

- `Conv2d`: 2D convolution forward pass and full backward pass (gradients for input, weights, and biases)
- `ReLU`: forward and backward
- `BatchNorm2d`: forward and backward with running statistics tracking
- Gradient checking to numerically verify manual gradients

## Dataset

MNIST (auto-downloaded via `torchvision` on first run).

## How to run

```bash
cd cnn/
jupyter notebook walkthrough.ipynb
```
