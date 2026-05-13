"""
VAE from scratch: Variational Autoencoder with manual implementation of core components.

Implements the full VAE pipeline:
- Gaussian log-likelihood computation
- KL divergence (exact closed-form for isotropic Gaussians)
- Reparameterization trick for differentiable sampling
- ELBO loss (reconstruction + KL regularization)
- Convolutional VAE architecture (encoder + decoder)
- Sample generation from the learned latent space

All probabilistic components are implemented from scratch without torch.distributions.

Originally developed as coursework for CPEN 455 (Deep Learning) at UBC,
refactored into a standalone module with docstrings and typed dimension annotations.
"""

import math

import torch
import torch.nn as nn


# -----------------------------------
# Core probabilistic building blocks
# -----------------------------------

def log_prob(
    x: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
) -> torch.Tensor:
    """
    Compute element-wise log-probability under an isotropic Gaussian.

    Given observations x, mean mu, and log standard deviation log_sigma,
    evaluates:

    log p(x | mu, sigma) = -K/2 log(2pi) - K log(sigma)
                            - 1/(2 sigma^2) sum_k (x_k - mu_k)^2

    where K is the dimensionality of each sample.

    Args:
        x:         (B, N, K): observations. B batches, N samples each, K dims.
        mu:        (B, N, K): means of the Gaussian.
        log_sigma: (B, N): log standard deviation (scalar sigma per sample).

    Returns:
        (B, N): log-probability for each sample in each batch.
    """
    sigma = log_sigma.exp()
    K = x.shape[-1]
    diff_sq = (x - mu).pow(2).sum(dim=-1)  # (B, N)

    # Three terms of the log-PDF:
    # -(K/2) log(2 pi)
    # -K log(sigma)
    # -(1 / 2 sigma^2) ||x - mu||^2
    term_const = -0.5 * K * math.log(2.0 * math.pi)
    term_sigma = -K * log_sigma
    term_exp = -0.5 * diff_sq / sigma.pow(2)

    return term_const + term_sigma + term_exp  # (B, N)


def kl_q_p_exact(
    params_q: torch.Tensor,
    params_p: torch.Tensor,
) -> torch.Tensor:
    """
    Exact KL divergence between two isotropic Gaussians.

    Uses the closed-form expression for KL(N_q || N_p) with isotropic
    covariance Sigma = sigma^2 I:

    KL = 0.5 [ K (sigma_q / sigma_p)^2
                + ||mu_q - mu_p||^2 / sigma_p^2
                - K
                + 2K log(sigma_p / sigma_q) ]

    Each params tensor packs [mu_1, ..., mu_K, log_sigma] along dim=-1.

    Args:
        params_q: (B, K+1): encoder distribution parameters.
        params_p: (B, K+1): prior (or target) distribution parameters.

    Returns:
        (B,): KL divergence for each element in the batch.
    """
    K = params_q.shape[-1] - 1

    mu_q, log_sig_q = params_q[:, :-1], params_q[:, -1]  # (B, K), (B,)
    mu_p, log_sig_p = params_p[:, :-1], params_p[:, -1]  # (B, K), (B,)

    sigma_q = log_sig_q.exp()
    sigma_p = log_sig_p.exp()

    variance_ratio = K * (sigma_q / sigma_p).pow(2)
    mean_diff = (mu_q - mu_p).pow(2).sum(dim=1) / sigma_p.pow(2)
    log_ratio = 2.0 * K * (log_sig_p - log_sig_q)

    return 0.5 * (variance_ratio + mean_diff - K + log_ratio)  # (B,)


def rsample(
    params_q: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """
    Sample from isotropic Gaussians via the reparameterization trick.

    Instead of sampling z ~ N(mu, sigma^2 I) directly (which blocks
    gradient flow), we sample epsilon ~ N(0, I) and compute:

    z = mu + sigma * epsilon

    This makes the sampling operation differentiable w.r.t. mu and sigma.

    Args:
        params_q:    (B, K+1): packed [mu_1..mu_K, log_sigma] parameters.
        num_samples: number of z samples to draw per batch element.

    Returns:
        (B, N, K): reparameterized samples, where N = num_samples.
    """
    B, K_plus_1 = params_q.shape
    K = K_plus_1 - 1

    mu = params_q[:, :-1]   # (B, K)
    log_sigma = params_q[:, -1]  # (B,)

    # epsilon ~ N(0, I): shape (B, N, K)
    eps = torch.randn(B, num_samples, K, device=params_q.device)

    # Broadcast: mu (B, 1, K), sigma (B, 1, 1)
    z = mu.unsqueeze(1) + log_sigma.exp().unsqueeze(1).unsqueeze(2) * eps

    return z  # (B, N, K)


def elbo_loss(
    x: torch.Tensor,
    params_q: torch.Tensor,
    num_samples: int,
    func_decoder,
    log_sig_x: float,
) -> torch.Tensor:
    """
    Compute the ELBO loss (negative ELBO) for a batch.

    ELBO = E_q(z|x)[ log p(x|z) ] - KL( q(z|x) || p(z) )

    We return the *negative* ELBO (the quantity to minimize):

    L = -E_q[ log p(x|z) ] + KL(q || p)
      = reconstruction_loss  + kl_term

    The reconstruction term is estimated via Monte Carlo with num_samples
    draws from q(z|x). The KL term uses the exact closed-form (both q and
    p are isotropic Gaussians; p(z) = N(0, I)).

    Args:
        x:            (B, D): flattened input data.
        params_q:     (B, K+1): encoder output [mu, log_sigma].
        num_samples:  number of z samples for the MC estimate.
        func_decoder: callable mapping (B, N, K) -> (B, N, D).
        log_sig_x:    scalar log std-dev of the decoder likelihood.

    Returns:
        (B,): per-sample ELBO loss (lower is better training signal).
    """
    z_q = rsample(params_q, num_samples)    # (B, N, K)
    mu_x = func_decoder(z_q)                # (B, N, D)
    B, N, D = mu_x.shape

    # Expand x to match sample dimension: (B, 1, D) -> (B, N, D)
    x_expanded = x.unsqueeze(1).expand(-1, N, -1)

    # log p(x | z) for each sample
    log_sig_x_tensor = x.new_full((B, N), log_sig_x)
    neg_recon = -log_prob(x_expanded, mu_x, log_sig_x_tensor)  # (B, N)
    recon_term = neg_recon.mean(dim=1)  # (B,) average over N samples

    # KL( q(z|x) || p(z) ),  p(z) = N(0, I) -> params_p = zeros
    params_p = torch.zeros_like(params_q)
    kl_term = kl_q_p_exact(params_q, params_p)  # (B,)

    return recon_term + kl_term  # (B,)


# ----------
# VAE model
# ----------

class VAE(nn.Module):
    """
    Convolutional Variational Autoencoder for MNIST (28x28 grayscale).

    Architecture:
    Encoder: Conv2d -> ReLU -> Conv2d -> ReLU -> Flatten -> Linear -> (mu, log_sigma)
    Decoder: Linear -> Unflatten -> ReLU -> ConvTranspose2d -> ReLU -> ConvTranspose2d

    The encoder outputs K+1 values: K means and 1 shared log-sigma for the
    isotropic Gaussian q(z|x). The decoder maps z back to image space and
    outputs the mean of p(x|z); a learnable scalar log_sig_x parameterizes
    the decoder variance.

    Args:
        latent_dim: dimensionality of the latent space z.
        num_filters: number of convolutional filters per layer.
    """

    def __init__(self, latent_dim: int = 2, num_filters: int = 32):
        super().__init__()

        self.latent_dim = latent_dim
        self.image_size = 28
        kernel_size = 5

        # Two valid (no-padding) convolutions each shrink spatial dims by
        # (kernel_size - 1), so: 28 -> 24 -> 20
        self.size_after_conv = self.image_size - 2 * 2 * (kernel_size // 2)
        self.flat_size = self.size_after_conv ** 2 * num_filters

        # --- Encoder ---
        # Input: (B, 1, 28, 28) -> Output: (B, K+1)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.flat_size, latent_dim + 1),
        )

        # --- Decoder ---
        # Input: (B, K) -> Output: (B, 1, 28, 28)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.flat_size),
            nn.Unflatten(1, (num_filters, self.size_after_conv, self.size_after_conv)),
            nn.ReLU(),
            nn.ConvTranspose2d(num_filters, num_filters, kernel_size),
            nn.ReLU(),
            nn.ConvTranspose2d(num_filters, 1, kernel_size),
        )

        # Learnable scalar decoder variance: log sigma_x
        self.log_sig_x = nn.Parameter(torch.zeros(()))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input images to distribution parameters.

        Args:
            x: (B, 1, 28, 28): batch of grayscale images.

        Returns:
            (B, K+1): packed [mu_1..mu_K, log_sigma].
        """
        return self.encoder(x)

    def decode(self, z_samples: torch.Tensor) -> torch.Tensor:
        """
        Decode latent samples to reconstructed image means.

        Args:
            z_samples: (B, N, K): latent samples (N samples per image).

        Returns:
            (B, N, 1, 28, 28): reconstructed image means.
        """
        B, N, K = z_samples.shape
        flat = z_samples.view(B * N, K)
        out = self.decoder(flat)  # (B*N, 1, 28, 28)
        return out.view(B, N, 1, self.image_size, self.image_size)

    def forward(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Run the full VAE and return the ELBO (higher = better).

        Computes: ELBO = E_q[ log p(x|z) ] - KL(q(z|x) || p(z))

        Args:
            x: (B, 1, 28, 28): input images.
            n_samples: number of z samples for MC reconstruction estimate.

        Returns:
            Scalar: mean ELBO across the batch (to be *maximized*).
        """
        phi = self.encode(x)                # (B, K+1)
        z = rsample(phi, n_samples)         # (B, N, K)
        mu_x = self.decode(z)               # (B, N, 1, 28, 28)

        B, C, H, W = x.shape
        x_flat = x.view(B, 1, -1).expand(-1, n_samples, -1)  # (B, N, C*H*W)
        mu_x_flat = mu_x.view(B, n_samples, -1)               # (B, N, C*H*W)

        log_sig = self.log_sig_x.view(1, 1).expand(B, n_samples)  # (B, N)

        recon = log_prob(x_flat, mu_x_flat, log_sig).mean()  # scalar
        kl = kl_q_p_exact(phi, torch.zeros_like(phi)).mean()  # scalar

        return recon - kl  # ELBO (maximize this)

    @torch.no_grad()
    def generate(self, n_images: int = 9) -> torch.Tensor:
        """
        Generate new images by sampling z ~ N(0, I) and decoding.

        During inference we have no input x, so we sample from the prior
        p(z) = N(0, I) and pass through the decoder.

        Args:
            n_images: number of images to generate.

        Returns:
            (n_images, 1, 28, 28): generated image means.
        """
        z = torch.randn(n_images, 1, self.latent_dim)  # (n_images, 1, K)
        # decode expects (B, N, K) and returns (B, N, 1, 28, 28)
        samples = self.decode(z)  # (n_images, 1, 1, 28, 28)
        return samples.squeeze(1)  # (n_images, 1, 28, 28)
    