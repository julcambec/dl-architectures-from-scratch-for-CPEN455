# Transformer from Scratch

An encoder-only Transformer implemented from first principles for binary sequence classification. Implementation at [transformer_from_scratch.py](transformer_from_scratch.py)

## Task

Given a random string over the alphabet `{c, p, e, n}`, classify whether it contains `"cpen"` as a contiguous substring. A `[CLS]` token is prepended; the model reads its output at position 0 to make the binary decision.

## Architecture

```
  Input: "ecpenp"
        │
        ▼
  ┌──────────────┐
  │  Tokenizer   │  Split into characters, prepend [CLS],
  │              │  convert to one-hot rows
  └──────┬───────┘
         │  (n+1) × 5 one-hot matrix
         ▼
  ┌──────────────┐
  │   Encoder    │  X · W_enc
  │              │  Project d_voc=5 → d_model
  └──────┬───────┘
         │  (n+1) × d_model
         ▼
  ┌──────────────┐
  │  Positional  │  APE: add learnable embeddings
  │  Encoding    │  RPE: bias added inside attention instead
  └──────┬───────┘
         │
         ▼
  ┌─────────────────────────────────────┐
  │        Transformer Layer  ×N        │
  │                                     │
  │  ┌───────────────────────────────┐  │
  │  │  Multi-Head Self-Attention    │  │
  │  │                               │  │
  │  │  Per head h:                  │  │
  │  │   Q_h = X·W_Q,h               │  │
  │  │   K_h = X·W_K,h               │  │
  │  │   V_h = X·W_V,h               │  │
  │  │                               │  │
  │  │   x =   Q_h · K_h^T [+ M_h]   │  │
  │  │        ─────────────────────  │  │
  │  │                √d_h           │  │
  │  │                               │  │
  │  │  attn = softmax(x)            │  │
  │  │                               │  │
  │  │  head_h = attn · V_h          │  │
  │  │                               │  │
  │  │  out = Concat(heads) · W_O    │  │
  │  └───────────────────────────────┘  │
  │         │                           │
  │         ├── + residual              │
  │         ├── LayerNorm               │
  │         ▼                           │
  │  ┌───────────────────────────────┐  │
  │  │  Feed-Forward Network         │  │
  │  │  ReLU(X · W_1) · W_2          │  │
  │  │  d_model → 4·d_model → d_model│  │
  │  └───────────────────────────────┘  │
  │         │                           │
  │         ├── + residual              │
  │         ├── LayerNorm               │
  │         ▼                           │
  └─────────────────────────────────────┘
         │  (n+1) × d_model
         ▼
  ┌──────────────┐
  │   Decoder    │  X · W_dec
  │              │  Project d_model → d_out=1
  └──────┬───────┘
         │  (n+1) × 1
         ▼
  ┌──────────────┐
  │  Extract     │  Take output at position 0
  │  [CLS]       │  (the prepended classification token)
  └──────┬───────┘
         │  scalar logit
         ▼
  ┌──────────────┐
  │  BCE Loss    │  sigmoid → binary prediction
  │              │  label ∈ {0, 1}
  └──────────────┘
```

## What's Implemented from Scratch

| Component | Description |
|---|---|
| `Tokenizer` | Character-level tokenizer producing one-hot matrices over a 5-token vocab |
| `AbsolutePositionalEncoding` | Learnable position embeddings added to token representations |
| `MultiHeadAttention` | Multi-head scaled dot-product self-attention with optional RPE |
| `TransformerLayer` | Self-attention + feed-forward block with residual connections and LayerNorm (Pre-Norm and Post-Norm) |
| `TransformerModel` | Full encoder: linear projection → positional encoding → N layers → linear decoder |
| `CustomScheduler` | Linear warmup + linear cooldown learning rate schedule |
| `Trainer` | Training loop with BCE loss on `[CLS]` output |

**What uses PyTorch:** `nn.Parameter`, `nn.LayerNorm`, `nn.ReLU`, `F.softmax`, `F.binary_cross_entropy_with_logits`, `Adam` optimizer, autograd for the outer training loop. All architecture internals (attention, projections, feed-forward, positional encoding) are manual.

## Key Equations

**Scaled dot-product attention (per head):**

$$\text{head}_h = \text{softmax}\!\left(\frac{(X W_{Q,h})(X W_{K,h})^\top}{\sqrt{d_h}}\right)(X W_{V,h})$$

**Multi-head output:**

$$\text{Attention}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) \cdot W_O$$

**Relative positional encoding (RPE):**

$$\text{head}_h = \text{softmax}\!\left(\frac{Q_h K_h^\top + M_h}{\sqrt{d_h}}\right) V_h$$

where $M_h$ is a learnable Toeplitz matrix with entry $(i,j) = m_{i-j,h}$, encoding relative distances with $2n - 1$ free parameters.

**Feed-forward network:**

$$\text{FC}(X) = \text{ReLU}(X W_1) \cdot W_2$$

where $W_1 \in \mathbb{R}^{d_{\text{model}} \times 4d_{\text{model}}}$, $W_2 \in \mathbb{R}^{4d_{\text{model}} \times d_{\text{model}}}$.

## APE vs. RPE

Both are implemented side by side.

**Absolute Positional Encoding (APE)** assigns a unique learnable vector to each position index. Simple and effective, but the encoding has no built-in notion of distance: position 3 and position 4 are as "different" as position 3 and position 30 unless the model learns otherwise.

**Relative Positional Encoding (RPE)** adds a learned scalar bias to each attention score based on the signed distance between the query and key positions. This makes the encoding translation-invariant: the model treats "tokens 2 positions apart" the same regardless of absolute position. RPE is implemented as a Toeplitz bias matrix inside the softmax, requiring only $2n - 1$ parameters per head.

## Result

The model achieves **~99–100% test accuracy** on the substring detection task (string length 16, 10k training samples, 5000 training steps).
