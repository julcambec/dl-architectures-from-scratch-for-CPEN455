# Variational Autoencoder (VAE), From Scratch

A convolutional VAE for MNIST with all probabilistic components implemented manually, without `torch.distributions`.

## What's Implemented from Scratch

| Component | Function / Class | What it computes |
|---|---|---|
| Gaussian log-likelihood | `log_prob()` | $\log p(\mathbf{x} \mid \boldsymbol{\mu}, \sigma^2 I) = -\frac{K}{2}\log(2\pi) - K\log\sigma - \frac{1}{2\sigma^2}\|\mathbf{x} - \boldsymbol{\mu}\|^2$ |
| Exact KL divergence | `kl_q_p_exact()` | $KL(\mathcal{N}_q \| \mathcal{N}_p) = \frac{1}{2}\left[K\frac{\sigma_q^2}{\sigma_p^2} + \frac{\|\boldsymbol{\mu}_q - \boldsymbol{\mu}_p\|^2}{\sigma_p^2} - K + 2K\log\frac{\sigma_p}{\sigma_q}\right]$ |
| Reparameterization trick | `rsample()` | $z = \boldsymbol{\mu} + \sigma \cdot \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(0, I)$ |
| ELBO loss | `elbo_loss()` | $\mathcal{L} = -\mathbb{E}_{q(z \mid x)}[\log p(x \mid z)] + KL(q(z \mid x) \| p(z))$ |
| VAE model | `VAE` class | Encoder, decoder, `forward()`, `generate()` |

## What Uses Standard PyTorch

The convolutional encoder and decoder use `nn.Conv2d`, `nn.ConvTranspose2d`, `nn.Linear`, `nn.ReLU`, and `nn.Sequential`. The optimizer (`torch.optim.Adam`) is also standard. The focus here is on the **probabilistic machinery**, the layers are not the main target.

## Key Concepts

**Reparameterization trick:** VAEs need to sample $z$ from the encoder distribution $q(z \mid x)$ during training, but sampling is not differentiable. The trick rewrites the sample as a deterministic function of learnable parameters plus fixed noise: $z = \mu + \sigma \cdot \varepsilon$, where $\varepsilon \sim \mathcal{N}(0, I)$. Gradients now flow through $\mu$ and $\sigma$ back to the encoder.

**ELBO:** The Evidence Lower Bound decomposes into a reconstruction term (how well the decoder recovers the input from $z$) and a KL regularization term (how close the encoder posterior stays to the standard normal prior). Without the KL term, the model degenerates into a deterministic autoencoder with no useful latent structure.

**Isotropic Gaussian simplification:** The encoder outputs a single shared $\log\sigma$ for all $K$ latent dimensions, rather than per-dimension variances. This reduces encoder output from $2K$ to $K+1$ values and keeps the KL closed-form simple. Diagonal-covariance encoders are more common in practice, as explained in the lectures.

## Architecture

```
Input (1, 28, 28)
    │
    ▼
 Encoder:  Conv2d(1→32, k=5) → ReLU → Conv2d(32→32, k=5) → ReLU → Flatten → Linear → [μ, log σ]
    │
    ▼
 Reparameterize:  z = μ + σ · ε,   ε ~ N(0, I)
    │
    ▼
 Decoder:  Linear → Unflatten → ReLU → ConvTranspose2d(32→32, k=5) → ReLU → ConvTranspose2d(32→1, k=5)
    │
    ▼
 Reconstruction (1, 28, 28)
```

## How to Run

```bash
# From repo root:
cd vae

# Run the walkthrough notebook (trains on MNIST, ~3-4 min on CPU)
jupyter notebook walkthrough.ipynb

```

MNIST auto-downloads via torchvision on first run (~12 MB).

## Results

With `latent_dim=2`, 10 epochs, `batch_size=256`, and `n_samples=1`:

- The model generates **recognizable but blurry** MNIST digits; expected for a basic VAE with Gaussian decoder.
- The 2D latent space shows smooth digit clustering with partial overlap between visually similar digits.
- Reconstruction quality confirms the encoder-decoder pipeline captures digit identity, though fine detail is lost.
