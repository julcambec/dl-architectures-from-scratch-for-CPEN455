# CNN From Scratch

Manual forward and backward passes for a convolutional neural network, no `loss.backward()`, no autograd for layer gradients. Implementation at [cnn_from_scratch.py](cnn_from_scratch.py)

## Architecture

```
Input (B, 1, 28, 28)
  │
  ├─ Conv2d(1 → 2, 3×3, pad=1)     im2col forward, vectorised im2col backward
  ├─ BatchNorm2d                   per-channel normalisation (γ=1, β=0)
  ├─ ReLU
  │   → (B, 2, 28, 28)
  │
  ├─ Conv2d(2 → 2, 3×3, pad=1)
  ├─ BatchNorm2d
  ├─ ReLU
  │   → (B, 2, 28, 28)
  │
  ├─ Flatten → (B, 1568)
  └─ Linear(1568, 10) → logits
```

Trained on MNIST (5 000-image subset) with manual SGD.

## What's Implemented From Scratch

| Component | Forward | Backward | Notes |
|---|---|---|---|
| `Conv2d` | im2col patch extraction → batch matmul | Vectorised: `grad_out @ im2col` for filter grad, col2im scatter for input grad | Caches im2col matrix from forward for reuse in backward |
| `BatchNorm2d` | Per-channel mean/var normalisation | Chain rule through mean and variance (three-term gradient) | γ=1, β=0 (no learnable scale/shift) |
| `ReLU` | Element-wise max(x, 0) | Hard gating: grad passes where x > 0 | |
| `CNN` | Composes all layers + linear readout | Full chain-rule backward through all layers | |
| `gradient_check` | — | Central finite differences for numerical verification | Used during development to validate analytical gradients |

**What uses PyTorch:** tensor operations (`torch.zeros`, `@`, `torch.bmm`, slicing), `torch.softmax` for cross-entropy gradient, `torchvision` for MNIST loading. No `nn.Module`, no `autograd`, no `torch.optim`.

## Key Equations

**Conv2d forward (im2col):**

$$y[b, d, i, j] = \sum_{c,p,q} \text{filter}[d, c, p, q] \cdot x_{\text{pad}}[b, c, \, iS{+}p, \, jS{+}q]$$

Equivalently: reshape all receptive-field patches into an im2col matrix $\mathbf{X} \in \mathbb{R}^{H'W' \times CKK}$, then $\mathbf{Y} = \mathbf{F} \cdot \mathbf{X}^\top$ where $\mathbf{F} \in \mathbb{R}^{D \times CKK}$ is the reshaped filter.

**Conv2d backward (vectorised):**

$$\frac{\partial L}{\partial \mathbf{F}} = \sum_b \; \text{grad\_out}_b \;\cdot\; \text{im2col}_b \qquad \frac{\partial L}{\partial \mathbf{X}} = \text{col2im}\!\left(\text{grad\_out}_b^\top \;\cdot\; \mathbf{F}\right)$$

**Batch normalisation:**

$$\hat{x}_{b,c,h,w} = \frac{x_{b,c,h,w} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}$$

where $\mu_c$ and $\sigma_c^2$ are computed over the $(B, H, W)$ dimensions for each channel $c$.

## About the Backward Pass

The Conv2d backward was initially implemented as a 7-deep nested Python loop (iterating over B, D, H\_out, W\_out, C, K, K) to make the element-wise gradient accumulation explicit. This version was verified against both PyTorch autograd and finite-difference gradient checking (see `gradient_check()` in the module).

The loop-based backward took ~56 seconds per batch of 32 on 28×28 images, far too slow for training. The final implementation replaces those loops with the same im2col matrix used in the forward pass: the filter gradient becomes a batch matmul (`torch.bmm`) and the input gradient uses a col2im scatter with only H\_out × W\_out = 784 loop iterations (batch-vectorised). The result: ~0.04 seconds per batch: a ~1 400× speedup, with numerically identical gradients (differences at the 1e-14 level).

## Files

```
cnn/
├── README.md                ← you are here
├── cnn_from_scratch.py      ← Conv2d, ReLU, BatchNorm2d, CNN, gradient_check
└── walkthrough.ipynb        ← verification + MNIST training + results
```
