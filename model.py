import torch
import torch.nn as nn
from modules import AMFModule, BlockwiseAttention, ProjectionLayer, TemporalSignificanceFilter


class CMTBlock(nn.Module):
    """This is a lightweight CMT implementation for reproducibility and smoke testing.

    This stable implementation applies temporal self-attention and an FFN while
    preserving the [batch, time, hidden] shape. It intentionally avoids the
    invalid channel-attention transpose used by the original prototype.
    """

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.temp_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        normalized = self.norm1(x)
        attended, _ = self.temp_attn(normalized, normalized, normalized)
        x = x + attended
        return x + self.ffn(self.norm2(x))


class AMCModel(nn.Module):
    """Compact, inspectable implementation of AMC = TAM + AMF + CMTs + TSF."""

    def __init__(
        self,
        v_dim,
        a_dim,
        t_dim,
        hid_dim=512,
        num_heads=8,
        dropout=0.1,
        amf_temperature=0.7,
        block_size=2,
        num_cmt_blocks=4,
    ):
        super().__init__()
        self.v_enc = ProjectionLayer(v_dim, hid_dim, dropout)
        self.a_enc = ProjectionLayer(a_dim, hid_dim, dropout)
        self.t_enc = ProjectionLayer(t_dim, hid_dim, dropout)

        self.tam = BlockwiseAttention(hid_dim, num_heads, block_size, dropout)
        self.amf = AMFModule(hid_dim, amf_temperature)
        self.cmts = nn.Sequential(
            *[CMTBlock(hid_dim, num_heads, dropout) for _ in range(num_cmt_blocks)]
        )
        self.tsf = TemporalSignificanceFilter(hid_dim)

    def forward(self, v, a, t):
        """Fuse aligned visual v, acoustic/music-context a, and text t."""
        v = self.v_enc(v)
        a = self.tam(self.a_enc(a))
        t = self.t_enc(t)

        v_r, a_r, t_r = self.amf(v, a, t)
        x = self.cmts(v_r + a_r + t_r)
        g_music = self.tsf(x)
        return x, g_music

    def encode_text_global(self, t):
        return self.tsf(self.t_enc(t))
