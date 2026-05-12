"""
Transformer architecture implemented from scratch for binary sequence classification.

Task: detect whether a string over the alphabet {c, p, e, n} contains "cpen" as a
contiguous substring. A [CLS] token is prepended; the model's prediction comes from
the [CLS] output position.

All core components (tokenizer, positional encodings, multi-head attention,
transformer layers, classifier) are implemented without using torch.nn convenience
modules for the architecture internals. Only nn.Parameter, nn.ParameterList,
nn.ModuleList, nn.LayerNorm, nn.ReLU, and functional utilities (F.softmax,
F.binary_cross_entropy_with_logits) are used.

Originally developed as a solution for an assignment for CPEN 455 (Deep Learning)
at UBC. Refactored into a standalone module with docstrings and dimension annotations.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader


# ---------
# Dataset
# ---------

class SubstringDataset(Dataset):
    """
    Synthetic dataset of random strings over {c, p, e, n}.

    Each string has a binary label:
        1 if the string contains "cpen" as a contiguous substring, 0 otherwise.

    The dataset is balanced: even-indexed samples have label 0, odd-indexed have label 1.
    A fixed NumPy RNG seed ensures reproducibility.
    """

    LETTERS = list("cpen")

    def __init__(self, seed: int, dataset_size: int, str_len: int = 20):
        super().__init__()
        self.str_len = str_len
        self.dataset_size = dataset_size
        self.rng = np.random.default_rng(seed)
        self.strings, self.labels = self._create_dataset()

    def __getitem__(self, index):
        return self.strings[index], self.labels[index]

    def __len__(self):
        return self.dataset_size

    def _create_dataset(self):
        strings, labels = [], []
        for i in range(self.dataset_size):
            label = i % 2
            string = self._generate_random_string(bool(label))
            strings.append(string)
            labels.append(label)
        return strings, labels

    def _generate_random_string(self, has_cpen: bool) -> str:
        while True:
            st = "".join(self.rng.choice(self.LETTERS, size=self.str_len))
            if ("cpen" in st) == has_cpen:
                return st


# ----------
# Tokenizer
# ----------

class Tokenizer:
    """
    Character-level tokenizer with a 5-token vocabulary.

    Vocabulary: [CLS]=0, c=1, p=2, e=3, n=4.

    Converts a raw string into a one-hot matrix of shape (seq_len, d_voc)
    where d_voc = 5. Optionally prepends a [CLS] token, making the output
    shape (seq_len + 1, d_voc).
    """

    def __init__(self) -> None:
        self.vocab = {"[CLS]": 0, "c": 1, "p": 2, "e": 3, "n": 4}

    def tokenize_string(self, string: str, add_cls_token: bool = True) -> torch.Tensor:
        """
        Convert a string to a one-hot matrix.

        Args:
            string: Raw input string over {c, p, e, n}.
            add_cls_token: If True, prepend [CLS] token.

        Returns:
            one_hot: Tensor of shape (n_tokens, d_voc) where
                     n_tokens = len(string) + 1 if add_cls_token else len(string),
                     d_voc = 5.
        """
        tokens = list(string)
        if add_cls_token:
            tokens = ["[CLS]"] + tokens
        indices = [self.vocab[t] for t in tokens]

        # Build one-hot matrix: each row is a one-hot vector over the vocabulary
        one_hot = torch.zeros((len(indices), len(self.vocab)))  # (n_tokens, d_voc)
        for pos, idx in enumerate(indices):
            one_hot[pos, idx] = 1.0
        return one_hot

    def tokenize_string_batch(
        self, strings: list[str], add_cls_token: bool = True
    ) -> torch.Tensor:
        """
        Tokenize a batch of strings.

        Returns:
            batch: Tensor of shape (B, n_tokens, d_voc).
        """
        return torch.stack(
            [self.tokenize_string(s, add_cls_token=add_cls_token) for s in strings],
            dim=0,
        )


# ----------------------
# Positional Encodings
# ----------------------

class AbsolutePositionalEncoding(nn.Module):
    """
    Learnable absolute positional encoding (APE).

    Maintains a learnable weight matrix W_pos of shape (max_len, d_model).
    The i-th row is the positional embedding for position i. Applied by
    element-wise addition to the input sequence:

    out_i = x_i + W_pos[i]

    Reference: Section 3.5 of Vaswani et al. (2017), "Attention Is All You Need",
    but using learned embeddings instead of fixed sinusoidal vectors.
    """

    MAX_LEN = 256

    def __init__(self, d_model: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty((self.MAX_LEN, d_model)))  # (max_len, d_model)
        nn.init.normal_(self.W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional embeddings to the input.

        Args:
            x: Input tensor of shape (B, N, d_model).

        Returns:
            out: Tensor of shape (B, N, d_model) with positional info added.
        """
        N = x.shape[1]
        # Slice only the first N positions; unsqueeze for batch broadcasting
        return x + self.W[:N, :].unsqueeze(0)  # (B, N, d_model)


# ---------------------
# Multi-Head Attention
# ---------------------

class MultiHeadAttention(nn.Module):
    """
    Multi-head scaled dot-product attention with optional relative positional encoding.

    For each head h:
        head_h = softmax( (X_Q W_Q,h)(X_K W_K,h)^T [+ M_h] / sqrt(d_h) ) (X_V W_V,h)

    where d_h = d_model / n_heads and M_h is an optional learnable Toeplitz matrix
    encoding relative positions (RPE). Heads are concatenated and projected:

    Attention(X_K, X_Q, X_V) = Concat(head_1, ..., head_H) W_O

    For self-attention, X_K = X_Q = X_V = X.

    RPE detail: M_h is a Toeplitz matrix with 2n-1 degrees of freedom,
    where entry (i,j) = m_{i-j,h}. This encodes only the *relative* distance
    between token positions i and j, making the encoding translation-invariant.

    Reference: Vaswani et al. (2017), "Attention Is All You Need", Section 3.2.
    """

    MAX_LEN = 256

    def __init__(self, d_model: int, n_heads: int, rpe: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_h = d_model // n_heads
        self.rpe = rpe

        # Per-head projection matrices: W_Q, W_K, W_V each (d_model, d_h)
        self.Wq = nn.ParameterList(
            [nn.Parameter(torch.empty((d_model, self.d_h))) for _ in range(n_heads)]
        )
        self.Wk = nn.ParameterList(
            [nn.Parameter(torch.empty((d_model, self.d_h))) for _ in range(n_heads)]
        )
        self.Wv = nn.ParameterList(
            [nn.Parameter(torch.empty((d_model, self.d_h))) for _ in range(n_heads)]
        )
        # Output projection: (d_model, d_model)
        self.Wo = nn.Parameter(torch.empty((d_model, d_model)))

        if rpe:
            # RPE parameters: 2*MAX_LEN+1 values covering offsets -(MAX_LEN)..+(MAX_LEN)
            self.rpe_w = nn.ParameterList(
                [nn.Parameter(torch.empty((2 * self.MAX_LEN + 1,))) for _ in range(n_heads)]
            )

        # Initialize weights
        for h in range(self.n_heads):
            nn.init.xavier_normal_(self.Wk[h])
            nn.init.xavier_normal_(self.Wq[h])
            nn.init.xavier_normal_(self.Wv[h])
            if rpe:
                nn.init.normal_(self.rpe_w[h])
        nn.init.xavier_normal_(self.Wo)

    def forward(
        self, key: torch.Tensor, query: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute multi-head attention.

        Args:
            key:   (B, N, d_model)
            query: (B, N, d_model)
            value: (B, N, d_model)

        Returns:
            out: (B, N, d_model)
        """
        B, N, D = query.shape
        heads_out = []

        for h in range(self.n_heads):
            Q_h = query.matmul(self.Wq[h])   # (B, N, d_h)
            K_h = key.matmul(self.Wk[h])     # (B, N, d_h)
            V_h = value.matmul(self.Wv[h])   # (B, N, d_h)

            # Raw attention scores: (B, N, N)
            scores = torch.bmm(Q_h, K_h.transpose(1, 2))

            # Optionally add relative positional encoding bias
            if self.rpe:
                # Build relative position offset matrix: entry (i,j) = i - j
                positions = torch.arange(N, device=scores.device)
                rel_offsets = positions.unsqueeze(1) - positions.unsqueeze(0)  # (N, N)
                rel_offsets += self.MAX_LEN  # shift to non-negative indices
                M_h = self.rpe_w[h][rel_offsets]  # (N, N) Toeplitz bias
                scores = scores + M_h.unsqueeze(0)  # broadcast over batch

            # Scale and softmax
            scores = scores / math.sqrt(self.d_h)
            attn_weights = F.softmax(scores, dim=-1)  # (B, N, N)

            head_out = torch.bmm(attn_weights, V_h)  # (B, N, d_h)
            heads_out.append(head_out)

        # Concatenate heads and project
        concat = torch.cat(heads_out, dim=2)  # (B, N, d_model)
        out = concat.matmul(self.Wo)           # (B, N, d_model)
        return out


# ------------------
# Transformer Layer
# ------------------

class TransformerLayer(nn.Module):
    """
    Single Transformer encoder layer: self-attention + feed-forward, with residual
    connections and layer normalization.

    The feed-forward network is:
    FC(X) = ReLU(X W_1) W_2
    where W_1 ∈ R^{d_model × 4*d_model}, W_2 ∈ R^{4*d_model × d_model}.

    Supports Pre-Norm (normalize before sublayer) and Post-Norm (normalize after)
    variants, controlled by the `prenorm` flag.

    Pre-Norm (prenorm=True):
    x = x + Attention(LN(x))
    x = x + FC(LN(x))

    Post-Norm (prenorm=False):
    x = LN(x + Attention(x))
    x = LN(x + FC(x))

    Reference: Vaswani et al. (2017), Section 3.1 (Post-Norm);
    Xiong et al. (2020), "On Layer Normalization in the Transformer Architecture" (Pre-Norm).
    """

    def __init__(self, d_model: int, n_heads: int, prenorm: bool, rpe: bool):
        super().__init__()
        self.d_model = d_model
        self.prenorm = prenorm

        self.attention = MultiHeadAttention(d_model, n_heads, rpe=rpe)

        # Feed-forward weights: d_model -> 4*d_model -> d_model
        self.fc_W1 = nn.Parameter(torch.empty((d_model, 4 * d_model)))
        self.fc_W2 = nn.Parameter(torch.empty((4 * d_model, d_model)))
        self.relu = nn.ReLU()

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        nn.init.xavier_normal_(self.fc_W1)
        nn.init.xavier_normal_(self.fc_W2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through one Transformer layer.

        Args:
            x: (B, N, d_model)

        Returns:
            out: (B, N, d_model)
        """
        if self.prenorm:
            # Pre-Norm: normalize before each sublayer
            normed = self.ln1(x)
            x = x + self.attention(normed, normed, normed)

            normed = self.ln2(x)
            x = x + self.relu(normed.matmul(self.fc_W1)).matmul(self.fc_W2)
        else:
            # Post-Norm: normalize after each sublayer
            x = self.ln1(x + self.attention(x, x, x))

            ff_out = self.relu(x.matmul(self.fc_W1)).matmul(self.fc_W2)
            x = self.ln2(x + ff_out)

        return x


# --------------
# Configuration
# --------------

class ModelConfig:
    """
    Configuration for TransformerModel.

    Attributes:
        n_layers:      Number of stacked Transformer layers.
        input_dim:     Dimensionality of input tokens (d_voc = 5).
        d_model:       Hidden dimension throughout the Transformer.
        n_heads:       Number of attention heads per layer.
        prenorm:       If True, use Pre-Norm; otherwise Post-Norm.
        pos_enc_type:  'ape' for absolute positional encoding,
                       'rpe' for relative positional encoding.
        output_dim:    Output dimension per token (1 for binary classification).
    """

    n_layers: int = 4
    input_dim: int = 5
    d_model: int = 256
    n_heads: int = 4
    prenorm: bool = True
    pos_enc_type: str = "ape"
    output_dim: int = 1

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            assert hasattr(self, k), f"Unknown config key: {k}"
            setattr(self, k, v)


# -------------------------------------------
# Transformer Model (Encoder-only classifier)
# -------------------------------------------

class TransformerModel(nn.Module):
    """Encoder-only Transformer for binary sequence classification.

    Architecture:
    1. Linear encoder:  W_enc ∈ R^{d_input × d_model} projects one-hot tokens
    into the model's hidden dimension.
    2. Positional encoding: APE (additive learned embeddings) or RPE (relative
    bias inside attention).
    3. N stacked TransformerLayers.
    4. Linear decoder:  W_dec ∈ R^{d_model × d_output} projects each token's
    representation to the output space.

    For classification, only the [CLS] token's output (position 0) is used downstream.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Encoder projection: (d_input, d_model)
        self.enc_W = nn.Parameter(torch.empty((cfg.input_dim, cfg.d_model)))

        # Positional encoding (APE is a separate module; RPE is inside attention)
        if cfg.pos_enc_type == "ape":
            self.ape = AbsolutePositionalEncoding(d_model=cfg.d_model)

        # Transformer layers
        self.transformer_layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    prenorm=cfg.prenorm,
                    rpe=(cfg.pos_enc_type == "rpe"),
                )
                for _ in range(cfg.n_layers)
            ]
        )

        # Decoder projection: (d_model, d_output)
        self.dec_W = nn.Parameter(torch.empty((cfg.d_model, cfg.output_dim)))

        nn.init.xavier_normal_(self.enc_W)
        nn.init.xavier_normal_(self.dec_W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the full Transformer.

        Args:
            x: One-hot input of shape (B, N, d_input).

        Returns:
            out: Logits of shape (B, N, d_output). Use position 0 ([CLS]) for
                 classification.
        """
        x = x.matmul(self.enc_W)  # (B, N, d_model)

        if self.cfg.pos_enc_type == "ape":
            x = self.ape(x)

        for layer in self.transformer_layers:
            x = layer(x)  # (B, N, d_model)

        out = x.matmul(self.dec_W)  # (B, N, d_output)
        return out


# -------------------------
# Learning Rate Scheduler
# -------------------------

class CustomScheduler(lr_scheduler._LRScheduler):
    """
    Linear warmup + linear cooldown learning rate scheduler.

    Learning rate profile:
    - Step 0:            lr = 0
    - Step warmup_steps: lr = base_lr  (peak)
    - Step total_steps:  lr = 0

    The warmup phase linearly increases from 0 to base_lr.
    The cooldown phase linearly decreases from base_lr back to 0.
    """

    def __init__(self, optimizer, total_steps: int, warmup_steps: int = 1000):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        """Compute the multiplier for the current step and apply to each param group."""
        t = self.last_epoch

        if t < self.warmup_steps:
            mult = float(t) / float(self.warmup_steps)
        elif t >= self.total_steps:
            mult = 0.0
        else:
            # Linear cooldown from 1.0 at warmup_steps to 0.0 at total_steps
            mult = 1.0 - float(t - self.warmup_steps) / float(
                self.total_steps - self.warmup_steps
            )

        return [group["initial_lr"] * mult for group in self.optimizer.param_groups]


# --------
# Trainer
# --------

class TrainerConfig:
    """Training hyperparameters."""

    lr: float = 0.003
    train_steps: int = 5000
    batch_size: int = 256
    evaluate_every: int = 100
    device: str = "cpu"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            assert hasattr(self, k), f"Unknown config key: {k}"
            setattr(self, k, v)


class Trainer:
    """
    Minimal training loop for the Transformer substring classifier.

    The loss is binary cross-entropy computed on the [CLS] token output (position 0).
    """

    def __init__(self, model: TransformerModel, cfg: TrainerConfig):
        self.cfg = cfg
        self.device = cfg.device
        self.tokenizer = Tokenizer()
        self.model = model.to(self.device)

    def train(self, train_dataset: SubstringDataset, val_dataset: SubstringDataset):
        """Run the training loop with periodic validation."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        scheduler = CustomScheduler(optimizer, self.cfg.train_steps)
        train_loader = DataLoader(
            train_dataset, shuffle=True, batch_size=self.cfg.batch_size
        )

        for step in range(self.cfg.train_steps):
            self.model.train()
            strings, y = next(iter(train_loader))
            x = self.tokenizer.tokenize_string_batch(strings)

            optimizer.zero_grad()
            loss, _ = self.compute_batch_loss_acc(x, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % self.cfg.evaluate_every == 0:
                val_loss, val_acc = self.evaluate_dataset(val_dataset)
                print(
                    f"Step {step}: Train Loss={loss.item():.4f}, "
                    f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
                )

    def compute_batch_loss_acc(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute binary cross-entropy loss and accuracy for one batch.

        Uses only the [CLS] token output (position 0 in the sequence dimension).

        Args:
            x: One-hot input, shape (B, N, d_input).
            y: Binary labels, shape (B,).

        Returns:
            loss: Scalar BCE loss.
            acc:  Scalar accuracy (fraction correct).
        """
        x = x.to(self.device)
        y = y.to(self.device).float()

        logits = self.model(x)          # (B, N, d_output)
        cls_logits = logits[:, 0, :]    # (B, d_output) — [CLS] position
        cls_logits = cls_logits.squeeze(-1)  # (B,)

        loss = F.binary_cross_entropy_with_logits(cls_logits, y)

        preds = (torch.sigmoid(cls_logits) > 0.5).float()
        acc = (preds == y).float().mean()
        return loss, acc

    @torch.no_grad()
    def evaluate_dataset(self, dataset: SubstringDataset) -> tuple[float, float]:
        """Evaluate the model on an entire dataset. Returns (avg_loss, avg_accuracy)."""
        self.model.eval()
        loader = DataLoader(dataset, shuffle=False, batch_size=self.cfg.batch_size)
        total_loss, total_acc, total_samples = 0.0, 0.0, 0

        for strings, y in loader:
            x = self.tokenizer.tokenize_string_batch(strings)
            loss, acc = self.compute_batch_loss_acc(x, y)
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_acc += acc.item() * batch_size
            total_samples += batch_size

        return total_loss / total_samples, total_acc / total_samples
    