import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import math

from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    Residual,
    PreNorm,
    LinearAttention,
)


class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon_length, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(inp_channels, out_channels, kernel_size),
                Conv1dBlock(out_channels, out_channels, kernel_size),
            ]
        )

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange("batch t -> batch t 1"),
        )

        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1)
            if inp_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t):
        """
        x : [ batch_size x inp_channels x horizon_length ]
        t : [ batch_size x embed_dim ]
        returns:
        out : [ batch_size x out_channels x horizon_length ]
        """
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class TemporalUnet(nn.Module):

    def __init__(
        self,
        horizon_length,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        attention=False,
        **kwargs,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"[ models/temporal ] Channel dimensions: {in_out}")

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_in, dim_out, embed_dim=time_dim, horizon_length=horizon_length
                        ),
                        ResidualTemporalBlock(
                            dim_out, dim_out, embed_dim=time_dim, horizon_length=horizon_length
                        ),
                        (
                            Residual(PreNorm(dim_out, LinearAttention(dim_out)))
                            if attention
                            else nn.Identity()
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                horizon_length = horizon_length // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(
            mid_dim, mid_dim, embed_dim=time_dim, horizon_length=horizon_length
        )
        self.mid_attn = (
            Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
            if attention
            else nn.Identity()
        )
        self.mid_block2 = ResidualTemporalBlock(
            mid_dim, mid_dim, embed_dim=time_dim, horizon_length=horizon_length
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_out * 2, dim_in, embed_dim=time_dim, horizon_length=horizon_length
                        ),
                        ResidualTemporalBlock(
                            dim_in, dim_in, embed_dim=time_dim, horizon_length=horizon_length
                        ),
                        (
                            Residual(PreNorm(dim_in, LinearAttention(dim_in)))
                            if attention
                            else nn.Identity()
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                horizon_length = horizon_length * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, cond, time):
        """
        x : [ batch x horizon_length x transition ]
        """

        x = einops.rearrange(x, "b h t -> b t h")

        t = self.time_mlp(time)
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            h_val = h.pop()
            x = torch.cat((x, h_val), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b t h -> b h t")
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, ff_mult=4, dropout=0.0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim)
        ff_dim = dim * ff_mult
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Time embedding projection
        self.time_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, x, t):
        """
        x: [batch_size, seq_len, dim]
        t: [batch_size, dim]
        """
        # Time embedding
        time_emb = self.time_proj(t).unsqueeze(1)  # [batch_size, 1, dim]
        
        # Apply time embedding
        x_time = x + time_emb
        
        # Self-attention with residual
        x_norm = self.norm1(x_time)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # Feedforward with residual
        x = x + self.ff(self.norm2(x))
        
        return x

class TemporalTransformer(nn.Module):
    def __init__(
        self,
        horizon_length,
        transition_dim,
        cond_dim,
        dim=128,
        depth=4,
        heads=4,
        dim_head=32,
        ff_mult=4,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        
        print(f"[ models/temporal ] Initializing TemporalTransformer with dim: {dim}, depth: {depth}, heads: {heads}")
        
        # Time embedding
        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )
        
        # Input projection
        self.input_proj = nn.Linear(transition_dim, dim)
        
        # Positional embedding
        self.register_buffer("pos_embedding", self._create_sinusoidal_positional_embedding(horizon_length, dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        # Optional condition embedding
        self.has_cond = cond_dim > 0
        if self.has_cond:
            self.cond_proj = nn.Sequential(
                nn.Linear(cond_dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim)
            )
        
        # Output projection
        self.output_proj = nn.Linear(dim, transition_dim)
        
        # Initialize model with approximately 100K-500K parameters for this size of input
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights for better convergence
        """
        for p in self.parameters():
            if p.dim() > 1:  # Skip LayerNorm and biases
                nn.init.xavier_uniform_(p)
    
    def _create_sinusoidal_positional_embedding(self, length, dim):
        """
        Create sinusoidal positional embeddings for transformer positions
        """
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pos_embedding = torch.zeros(length, dim)
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        return pos_embedding
        
    def forward(self, x, cond, time):
        """
        x: [batch_size, horizon_length, transition_dim]
        cond: conditional information [batch_size, cond_dim] or None
        time: [batch_size] time embedding
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Time embedding
        t = self.time_mlp(time)  # [batch_size, dim]
        
        # Project input to model dimension
        x = self.input_proj(x)  # [batch_size, horizon_length, dim]
        
        # Add positional embeddings
        x = x + self.pos_embedding.unsqueeze(0)  # [batch_size, horizon_length, dim]
        
        # Add conditional embedding if available
        if self.has_cond and cond is not None:
            cond_emb = self.cond_proj(cond).unsqueeze(1)  # [batch_size, 1, dim]
            x = x + cond_emb
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, t)
        
        # Project back to transition dimension
        x = self.output_proj(x)  # [batch_size, horizon_length, transition_dim]
        
        return x


