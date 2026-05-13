# Deep Learning Architectures from Scratch for CPEN455

Educational from-scratch implementations of three foundational deep learning architectures: a **CNN** with manual backpropagation, a **Transformer** with multi-head self-attention, and a **VAE** with the reparameterization trick. Each implementation exposes the core math, peeking at what happens behind `loss.backward()`.

## Summary

| Architecture | Task | Dataset | Key from-scratch components |
|---|---|---|---|
| [CNN](cnn/) | Digit classification | MNIST | Conv2d forward + backward, BatchNorm, ReLU gradient |
| [Transformer](transformer/) | Substring detection | Synthetic strings | Multi-head self-attention, APE/RPE, LR scheduler |
| [VAE](vae/) | Generative modeling | MNIST | Reparameterization trick, KL divergence, ELBO |

## What's from scratch vs. what uses PyTorch

All architecture internals (convolution operations, attention mechanisms, positional encodings, batch normalization, activation gradients, the reparameterization trick, and loss functions) are implemented manually. PyTorch is used only for tensor operations, automatic differentiation in the outer training loop (where applicable), `torch.optim` optimizers, and data loading utilities. The goal is to expose what these components do mathematically, not to avoid PyTorch entirely.

## Where is what

```
dl-architectures-from-scratch/
├── README.md
├── requirements.txt
├── cnn/
│   ├── cnn_from_scratch.py         # Conv2d, ReLU, BatchNorm with manual backward passes
│   └── walkthrough.ipynb           # Math → code → train on MNIST → results
├── transformer/
│   ├── transformer_from_scratch.py # Tokenizer, positional encodings, multi-head attention
│   └── walkthrough.ipynb           # Math → code → train on SubstringDataset → results
└── vae/
    ├── vae_from_scratch.py         # Encoder, Decoder, reparameterization, ELBO loss
    └── walkthrough.ipynb           # Math → code → train on MNIST → generated samples
```

## Origin

I originally developed these implementations as solved assignments for CPEN 455 (Deep Learning) at UBC and refined them into documented code.

## License

[MIT](LICENSE)
