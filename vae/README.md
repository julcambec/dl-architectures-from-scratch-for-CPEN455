# VAE from Scratch

A Variational Autoencoder with the **reparameterization trick and ELBO loss** implemented from first principles. Trained on MNIST for generative modeling and sample synthesis.

## What's implemented from scratch

- Encoder network (outputs mean and log-variance of the latent distribution)
- Reparameterization trick (enables gradient flow through stochastic sampling)
- Decoder network (reconstructs images from latent samples)
- ELBO loss function (reconstruction loss + KL divergence, with closed-form KL for Gaussians)

## Dataset

MNIST (auto-downloaded via `torchvision` on first run).

## How to run

```bash
cd vae/
jupyter notebook walkthrough.ipynb
```
