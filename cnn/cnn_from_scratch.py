"""
From-scratch CNN building blocks with manual forward and backward passes.

Implements 2D convolution (via im2col), ReLU, Batch Normalization, and a
simple CNN architecture, all with hand-written gradient computations rather
than relying on PyTorch autograd.

Key equations and design notes are in docstrings. For a full walkthrough
with training on MNIST, see ``walkthrough.ipynb``.

Originally developed as coursework for CPEN 455 (Deep Learning) at UBC;
refactored into a standalone importable module with classes, docstrings,
and device-agnostic tensor handling.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Conv2d — forward (im2col) and backward (vectorised im2col / col2im)
# ---------------------------------------------------------------------

class Conv2d:
    """
    2D convolution layer with manual forward and backward passes.

    Forward uses the **im2col** approach: extract every (C×K×K) patch from
    the padded input into columns of a matrix, then multiply by the reshaped
    filter matrix.  This converts the spatial sliding-window operation into
    a single matrix multiplication per image in the batch.

    Backward computes gradients using the same im2col representation:

    ∂L/∂filter  = Σ_b  grad_out_b  @  im2col_b
                     (D, H'W') @ (H'W', CKK)  →  (D, CKK)  →  (D, C, K, K)

    ∂L/∂input   = col2im( grad_out_b^T @ filter_rows )
                     Scatter (H'W', CKK) patches back to padded input shape.

    This vectorised formulation is mathematically identical to the
    element-wise nested-loop version (which was verified against PyTorch
    autograd during development) but runs ~100× faster by leveraging
    matrix multiplications instead of 7-deep Python loops.

    Assumptions: square kernel, equal stride in both dimensions, same
    padding on all sides.

    Parameters
    ----------
    in_channels : int
        Number of input channels (C).
    out_channels : int
        Number of filters / output channels (D).
    kernel_size : int
        Side length of the square kernel (K).  Default 3.
    stride : int
        Stride of the convolution.  Default 1.
    padding : int
        Zero-padding added to both sides of each spatial dimension.  Default 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Weights initialised by the caller (or by CNN class).
        # Shape: (D, C, K, K)
        self.weight: torch.Tensor | None = None

        # Cached during forward for use in backward.
        self._input: torch.Tensor | None = None
        self._im2col: torch.Tensor | None = None

    # ---- forward -----

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute 2D convolution via im2col.

        Parameters
        ----------
        x : Tensor, shape (B, C, H, W)

        Returns
        -------
        out : Tensor, shape (B, D, H_out, W_out)
            where H_out = (H + 2*padding - K) // stride + 1  (likewise W).
        """
        assert self.weight is not None, "Set .weight before calling forward()"
        self._input = x

        filt = self.weight
        K = self.kernel_size
        S = self.stride
        P = self.padding

        B, C, H, W = x.shape
        D = self.out_channels
        device = x.device
        dtype = x.dtype

        H_out = (H + 2 * P - K) // S + 1
        W_out = (W + 2 * P - K) // S + 1

        # --- pad input ---
        x_pad = torch.zeros(
            (B, C, H + 2 * P, W + 2 * P), dtype=dtype, device=device
        )
        x_pad[:, :, P : P + H, P : P + W] = x

        # --- build im2col matrix ---
        # Each row is a flattened (C*K*K) receptive-field patch.
        # Shape: (B, H_out*W_out, C*K*K)
        im2col = torch.zeros(
            (B, H_out * W_out, C * K * K), dtype=dtype, device=device
        )
        idx = 0
        for i in range(H_out):
            for j in range(W_out):
                patch = x_pad[
                    :, :, i * S : i * S + K, j * S : j * S + K
                ]  # (B, C, K, K)
                im2col[:, idx, :] = patch.reshape(B, -1)
                idx += 1

        self._im2col = im2col  # cache for backward

        # --- filter as row vectors: (D, C*K*K) ---
        filt_rows = filt.reshape(D, -1)

        # --- matmul: filt_rows @ im2col^T -> (B, D, H_out*W_out) ---
        out = torch.bmm(
            filt_rows.unsqueeze(0).expand(B, -1, -1),   # (B, D, CKK)
            im2col.transpose(1, 2),                       # (B, CKK, H'W')
        )  # -> (B, D, H'W')

        return out.reshape(B, D, H_out, W_out)

    # ---- backward ----

    def backward(
        self, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gradients of the loss w.r.t. input and filter weights.

        Uses the cached im2col matrix from ``forward()`` to compute both
        gradients via matrix multiplications (vectorised im2col backward),
        then scatters the input gradient back via col2im.

        Parameters
        ----------
        grad_out : Tensor, shape (B, D, H_out, W_out)
            Upstream gradient (∂L/∂y).

        Returns
        -------
        grad_input : Tensor, shape (B, C, H, W)
            ∂L/∂x — gradient w.r.t. the layer input.
        grad_weight : Tensor, shape (D, C, K, K)
            ∂L/∂filter — gradient w.r.t. the filter weights.
        """
        assert self._input is not None, "Call forward() before backward()"
        assert self._im2col is not None

        x = self._input
        filt = self.weight
        im2col = self._im2col

        K = self.kernel_size
        S = self.stride
        P = self.padding

        B, C, H, W = x.shape
        D = self.out_channels
        device = x.device
        dtype = x.dtype

        H_out = (H + 2 * P - K) // S + 1
        W_out = (W + 2 * P - K) // S + 1

        filt_rows = filt.reshape(D, -1)              # (D, CKK)
        grad_out_flat = grad_out.reshape(B, D, -1)   # (B, D, H'W')

        # --- ∂L/∂filter (vectorised) ---
        # For each image b:  (D, H'W') @ (H'W', CKK) -> (D, CKK)
        # Sum across batch.
        grad_weight = torch.bmm(
            grad_out_flat,   # (B, D, H'W')
            im2col,          # (B, H'W', CKK)
        ).sum(dim=0)        # (D, CKK)
        grad_weight = grad_weight.reshape(filt.shape)  # (D, C, K, K)

        # --- ∂L/∂input (vectorised via col2im) ---
        # Step 1: gradient in im2col space.
        # (H'W', D) @ (D, CKK) -> (H'W', CKK)  per image
        grad_im2col = torch.bmm(
            grad_out_flat.transpose(1, 2),                    # (B, H'W', D)
            filt_rows.unsqueeze(0).expand(B, -1, -1),         # (B, D, CKK)
        )  # -> (B, H'W', CKK)

        # Step 2: col2im — scatter patches back to the padded input grid.
        grad_x_pad = torch.zeros(
            (B, C, H + 2 * P, W + 2 * P), dtype=dtype, device=device
        )
        idx = 0
        for i in range(H_out):
            for j in range(W_out):
                patch_grad = grad_im2col[:, idx, :].reshape(B, C, K, K)
                grad_x_pad[
                    :, :, i * S : i * S + K, j * S : j * S + K
                ] += patch_grad
                idx += 1

        # Strip padding.
        grad_input = grad_x_pad[:, :, P : P + H, P : P + W]

        return grad_input, grad_weight


# ------------------------------
# ReLU — forward and backward
# ------------------------------

class ReLU:
    """
    Rectified Linear Unit: f(x) = max(x, 0).

    Backward: gradient passes through where x > 0, zeroed elsewhere.
    This "hard gating" is what causes the "dying ReLU" phenomenon,
    neurons whose inputs are always negative stop learning.
    """

    def __init__(self) -> None:
        self._input: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ReLU element-wise. Input/output: (B, C, H, W)."""
        self._input = x
        return x * (x > 0)

    def backward(self, grad_out: torch.Tensor) -> torch.Tensor:
        """∂L/∂x = grad_out * 1{x > 0}. Input/output: (B, C, H, W)."""
        assert self._input is not None, "Call forward() before backward()"
        return grad_out * (self._input > 0)


# ------------------------------------------------------
# BatchNorm2d — forward and backward  (gamma=1, beta=0)
# ------------------------------------------------------

class BatchNorm2d:
    """
    Batch Normalisation for convolutional feature maps (gamma=1, beta=0).

    Per-channel statistics computed over (B, H, W):

    mu[c]    = (1 / BHW) * Σ_{b,h,w}  x[b,c,h,w]
    var[c]   = (1 / BHW) * Σ_{b,h,w} (x[b,c,h,w] - mu[c])²
    y[b,c,h,w] = (x[b,c,h,w] - mu[c]) / sqrt(var[c] + eps)

    The backward pass differentiates through both the mean and the variance
    terms, which depend on *all* elements in the (B, H, W) slice.

    Parameters
    ----------
    epsilon : float
        Small constant for numerical stability in the denominator.
    """

    def __init__(self, epsilon: float = 1e-5) -> None:
        self.epsilon = epsilon
        self._input: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise each channel to zero mean / unit variance.  Shape: (B, C, H, W)."""
        self._input = x
        B, C, H, W = x.shape
        N = B * H * W
        mean = x.sum(dim=(0, 2, 3), keepdim=True) / N
        var = ((x - mean) ** 2).sum(dim=(0, 2, 3), keepdim=True) / N
        return (x - mean) / torch.sqrt(var + self.epsilon)

    def backward(self, grad_out: torch.Tensor) -> torch.Tensor:
        """Compute ∂L/∂x.  Shape: (B, C, H, W)."""
        assert self._input is not None, "Call forward() before backward()"
        x = self._input
        eps = self.epsilon

        B, C, H, W = x.shape
        N = B * H * W

        mean = x.sum(dim=(0, 2, 3), keepdim=True) / N
        var = ((x - mean) ** 2).sum(dim=(0, 2, 3), keepdim=True) / N
        inv_std = 1.0 / torch.sqrt(var + eps)

        grad_sum = grad_out.sum(dim=(0, 2, 3), keepdim=True)
        grad_xmu_sum = (grad_out * (x - mean)).sum(
            dim=(0, 2, 3), keepdim=True
        )

        grad_x = (
            grad_out
            - grad_sum / N
            - (x - mean) * grad_xmu_sum / (N * (var + eps))
        ) * inv_std

        return grad_x


# ---------------------------------------------------------
# CNN — two-layer Conv→BN→ReLU network with linear readout
# ---------------------------------------------------------

class CNN:
    """
    Simple CNN: Conv→BN→ReLU → Conv→BN→ReLU → Flatten → Linear (10 classes).

    All convolutions, batch normalisation, and ReLU layers use the manual
    forward/backward implementations above. Only the final cross-entropy
    loss computation uses ``torch.nn.functional``.

    Architecture
    ------------
    Layer 1:  Conv2d(C, D, K) → BatchNorm2d → ReLU
    Layer 2:  Conv2d(D, D, K) → BatchNorm2d → ReLU
    Readout:  Flatten → Linear(D*H*W, 10)

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for grayscale MNIST).
    num_filters : int
        Number of conv filters in each layer (D).
    kernel_size : int
        Square kernel side length (K).
    stride : int
    padding : int
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_filters: int = 2,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        self.conv1 = Conv2d(in_channels, num_filters, kernel_size, stride, padding)
        self.bn1 = BatchNorm2d()
        self.relu1 = ReLU()
        self.conv2 = Conv2d(num_filters, num_filters, kernel_size, stride, padding)
        self.bn2 = BatchNorm2d()
        self.relu2 = ReLU()
        self.linear_weight: torch.Tensor | None = None
        self._flat: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        filter_1: torch.Tensor,
        filter_2: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.  Input: (B, C, H, W) → Output: (B, 10) logits."""
        self.conv1.weight = filter_1
        self.conv2.weight = filter_2
        self.linear_weight = weight

        y = self.conv1.forward(x)
        y = self.bn1.forward(y)
        y = self.relu1.forward(y)

        y = self.conv2.forward(y)
        y = self.bn2.forward(y)
        y = self.relu2.forward(y)

        B = y.shape[0]
        self._flat = y.reshape(B, -1)       # (B, D*H*W)
        logits = self._flat @ weight        # (B, 10)
        return logits

    def backward(
        self, grad_loss: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Backward pass: gradients w.r.t. filter_1, filter_2, weight.

        Parameters
        ----------
        grad_loss : Tensor, shape (B, 10)
            Gradient of the loss w.r.t. the logits.

        Returns
        -------
        grad_filter_1 : Tensor, shape (D, C, K, K)
        grad_filter_2 : Tensor, shape (D, D, K, K)
        grad_weight   : Tensor, shape (D*H*W, 10)
        """
        assert self._flat is not None, "Call forward() before backward()"

        # --- linear readout ---
        grad_weight = self._flat.T @ grad_loss          # (D*H*W, 10)
        grad_flat = grad_loss @ self.linear_weight.T    # (B, D*H*W)

        # Reshape back to 4-D feature-map shape.
        shape_4d = self.bn2._input.shape
        grad_y6 = grad_flat.reshape(shape_4d)

        # --- layer 2 backward: ReLU → BN → Conv ---
        grad_y5 = self.relu2.backward(grad_y6)
        grad_y4 = self.bn2.backward(grad_y5)
        grad_y3, grad_filter_2 = self.conv2.backward(grad_y4)

        # --- layer 1 backward: ReLU → BN → Conv ---
        grad_y2 = self.relu1.backward(grad_y3)
        grad_y1 = self.bn1.backward(grad_y2)
        _, grad_filter_1 = self.conv1.backward(grad_y1)

        return grad_filter_1, grad_filter_2, grad_weight


# -------------------------------------------
# Utility: finite-difference gradient checker
# -------------------------------------------

def gradient_check(
    x: torch.Tensor,
    filt: torch.Tensor,
    grad_out: torch.Tensor,
    conv_fn=F.conv2d,
    epsilon: float = 1e-5,
    stride: int = 1,
    padding: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Verify analytical gradients via central finite differences.

    For a scalar loss  ℓ(h, x) = y^T v  where  y = conv(x, h):

    ∂ℓ/∂h[i] ≈ (ℓ(h + ε·e_i) − ℓ(h − ε·e_i)) / (2ε)

    This is slow (one forward pass per parameter element) but provides a
    ground-truth reference that doesn't depend on the correctness of the
    analytical backward implementation.

    Parameters
    ----------
    x : Tensor, shape (B, C, H, W)
    filt : Tensor, shape (D, C, K, K)
    grad_out : Tensor, shape (B, D, H_out, W_out)
        The "v" vector (same shape as conv output).
    conv_fn : callable
        Convolution function, default ``F.conv2d``.
    epsilon : float
    stride, padding : int

    Returns
    -------
    grad_x_fd : Tensor, shape (B, C, H, W)
    grad_filt_fd : Tensor, shape (D, C, K, K)
    """
    device = x.device
    B, C, H, W = x.shape
    D, C2, K, K2 = filt.shape

    # --- gradient w.r.t. filter ---
    grad_filt_fd = torch.zeros_like(filt)
    for d in range(D):
        for c in range(C2):
            for p in range(K):
                for q in range(K):
                    val = filt[d, c, p, q].item()

                    filt_up = filt.clone()
                    filt_up[d, c, p, q] = val + epsilon
                    loss_p = (
                        conv_fn(x, filt_up, stride=stride, padding=padding)
                        * grad_out
                    ).sum()

                    filt_dn = filt.clone()
                    filt_dn[d, c, p, q] = val - epsilon
                    loss_m = (
                        conv_fn(x, filt_dn, stride=stride, padding=padding)
                        * grad_out
                    ).sum()

                    grad_filt_fd[d, c, p, q] = (loss_p - loss_m) / (
                        2.0 * epsilon
                    )

    # --- gradient w.r.t. input ---
    grad_x_fd = torch.zeros_like(x)
    for b in range(B):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    val = x[b, c, h, w].item()

                    x_up = x.clone()
                    x_up[b, c, h, w] = val + epsilon
                    loss_p = (
                        conv_fn(x_up, filt, stride=stride, padding=padding)
                        * grad_out
                    ).sum()

                    x_dn = x.clone()
                    x_dn[b, c, h, w] = val - epsilon
                    loss_m = (
                        conv_fn(x_dn, filt, stride=stride, padding=padding)
                        * grad_out
                    ).sum()

                    grad_x_fd[b, c, h, w] = (loss_p - loss_m) / (
                        2.0 * epsilon
                    )

    return grad_x_fd, grad_filt_fd
