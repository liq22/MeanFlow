import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from einops import repeat


def modulate(x, scale, shift):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbed1D(nn.Module):
    def __init__(self, input_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = input_size // patch_size
        self.patch_size = patch_size

    def forward(self, x):
        x = self.proj(x)  # N, dim, T'
        x = x.transpose(1, 2)  # N, T', dim
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, nfreq=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(nfreq, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.nfreq = nfreq

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t = t * 1000
        t_freq = self.timestep_embedding(t, self.nfreq)
        return self.mlp(t_freq)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.g


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), scale_msa, shift_msa).transpose(0,1),
                                                  modulate(self.norm1(x), scale_msa, shift_msa).transpose(0,1),
                                                  modulate(self.norm1(x), scale_msa, shift_msa).transpose(0,1))[0].transpose(0,1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), scale_mlp, shift_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size, out_dim):
        super().__init__()
        self.norm_final = RMSNorm(dim)
        self.linear = nn.Linear(dim, patch_size * out_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.patch_size = patch_size
        self.out_dim = out_dim

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class TSDiT(nn.Module):
    def __init__(self, input_size=1024, patch_size=16, in_channels=2, dim=256, depth=8, num_heads=8, num_classes=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_classes = num_classes

        self.x_embedder = PatchEmbed1D(input_size, patch_size, in_channels, dim)
        self.t_embedder = TimestepEmbedder(dim)
        self.r_embedder = TimestepEmbedder(dim)
        self.use_cond = num_classes is not None
        self.y_embedder = nn.Embedding(num_classes + 1, dim) if self.use_cond else None

        self.pos_embed = nn.Parameter(torch.zeros(1, self.x_embedder.num_patches, dim), requires_grad=True)

        self.blocks = nn.ModuleList([DiTBlock(dim, num_heads) for _ in range(depth)])
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic_init)
        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.weight, std=0.02)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        N, T, D = x.shape
        x = x.reshape(N, T, self.patch_size, self.out_channels)
        x = x.permute(0,3,1,2).reshape(N, self.out_channels, T*self.patch_size)
        return x

    def forward(self, x, t, r, y=None):
        L = x.shape[-1]
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        r = self.r_embedder(r)
        c = t + r
        if self.use_cond:
            y = self.y_embedder(y)
            c = c + y
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x
