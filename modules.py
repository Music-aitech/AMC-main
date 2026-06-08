import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, feature] input, got {tuple(x.shape)}")
        return self.net(x)


class BlockwiseAttention(nn.Module):
    """Temporal Alignment Module (TAM) using local blockwise attention."""

    def __init__(self, dim, heads=8, block_size=2, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )

    def forward(self, x):
        b, n, d = x.shape
        pad_len = (self.block_size - n % self.block_size) % self.block_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        num_blocks = x.shape[1] // self.block_size
        x_blocked = x.reshape(b * num_blocks, self.block_size, d)
        normalized = self.norm(x_blocked)
        attn_out, _ = self.attn(normalized, normalized, normalized)
        out = (x_blocked + attn_out).reshape(b, -1, d)
        return out[:, :n, :]


class AMFModule(nn.Module):
    """Adaptive Multimodal Fusion with reliability-aware modality weights."""

    def __init__(self, dim, temperature=0.7):
        super().__init__()
        if temperature <= 0:
            raise ValueError("AMF temperature must be positive")
        self.temperature = temperature
        self.phi = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, v, a, t):
        if v.shape != a.shape or v.shape != t.shape:
            raise ValueError(
                "AMF expects aligned v/a/t sequences with identical shapes; "
                f"got v={tuple(v.shape)}, a={tuple(a.shape)}, t={tuple(t.shape)}"
            )
        modalities = torch.stack([v, a, t], dim=1)  # [B, 3, T, D]
        scores = self.phi(modalities)  # [B, 3, T, 1]
        weights = F.softmax(scores / self.temperature, dim=1)
        rectified = modalities * weights
        return rectified.unbind(dim=1)


class TemporalSignificanceFilter(nn.Module):
    """Temporal Semantic Fusion (TSF) used to form the training embedding."""

    def __init__(self, dim):
        super().__init__()
        self.alpha_net = nn.Linear(dim, 1)

    def forward(self, x):
        weights = F.softmax(self.alpha_net(x), dim=1)  # [B, T, 1]
        g = torch.sum(weights * x, dim=1)
        return g
