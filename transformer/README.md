# Transformer from Scratch

A Transformer classifier with **multi-head self-attention, absolute and relative positional encodings (APE/RPE), and a full training pipeline** — all implemented from first principles. Trained on a synthetic substring detection task.

## What's implemented from scratch

- Character-level tokenizer and one-hot encoding
- Absolute Positional Encoding (APE) and Relative Positional Encoding (RPE)
- Scaled dot-product attention
- Multi-head self-attention
- Transformer layer (attention + feed-forward + layer norm + residual connections)
- Classification head with custom learning rate scheduler

## Dataset

Synthetic `SubstringDataset`: random strings over the alphabet `{c, p, e, n}`, labeled by whether they contain the substring `"cpen"`. Generated in-code — no download needed.

## How to run

```bash
cd transformer/
jupyter notebook walkthrough.ipynb
```
