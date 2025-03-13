import torch
import torch.nn as nn
from torch.nn import functional as F

import warnings

warnings.formatwarning = lambda message, category, filename, lineno, line=None: \
    f'{category.__name__}: {message}\n'


class PatchEmbed3D(nn.Module):
    """Converts 3D input into patch embeddings."""
    def __init__(self, in_channels, embed_dim, patch_size=(2, 4, 4)):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, C, T, H, W) -> (B, D, T//pT, H//pH, W//pW)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x
    

class ViTBlock(nn.Module):
    """Basic Vision Transformer block."""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class CausalGroupNorm(nn.GroupNorm):
    """
    A GroupNorm that does not use temporal statistics, to ensure causality
    """
    def __init__(self, num_groups, num_channels, **kwargs):
        super().__init__(num_groups, num_channels, **kwargs)
        
    def forward(self, input):
        x = input.moveaxis(1, 2)  # (B, T, C, H, W)
        x_shape = x.shape
        x = x.flatten(0, 1)  # (B * T, C, H, W)
        x = super().forward(x).reshape(x_shape)
        return x.moveaxis(1, 2)  # (B, C, T, H, W)


act_layer = lambda: nn.ReLU()
bn_block = lambda features: nn.Sequential(nn.BatchNorm3d(features), act_layer())
gn_block = lambda features: nn.Sequential(CausalGroupNorm(4, features), act_layer())


class SpatialBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 embed_dim=128, 
                 num_heads=4, 
                 norms='mixed'):
        super().__init__()
        self.norms = norms
        self.patch_embed = PatchEmbed3D(in_channels, embed_dim)
        self.vit_block = ViTBlock(embed_dim, num_heads)
        self.proj = nn.Linear(embed_dim, out_channels)

    def forward(self, input):
        x = self.patch_embed(input)
        x = self.vit_block(x)
        x = self.proj(x).transpose(1, 2).view(input.shape[0], -1, *input.shape[2:])
        return x


class TemporalBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 embed_dim=128, 
                 num_heads=4, 
                 norms='mixed'):
        super().__init__()
        self.norms = norms
        self.patch_embed = PatchEmbed3D(in_channels, embed_dim, patch_size=(3, 1, 1))
        self.vit_block = ViTBlock(embed_dim, num_heads)
        self.proj = nn.Linear(embed_dim, out_channels)

    def forward(self, input):
        x = self.patch_embed(input)
        x = self.vit_block(x)
        x = self.proj(x).transpose(1, 2).view(input.shape[0], -1, *input.shape[2:])
        return x


class TennStViT(nn.Module):
    def __init__(
        self, 
        channels, 
        t_kernel_size, 
        n_depthwise_layers, 
        detector_head, 
        detector_depthwise, 
        norms='mixed',
    ):
        super().__init__()
        self.detector = detector_head
        
        depthwises = [False] * (10 - n_depthwise_layers) + [True] * n_depthwise_layers
        temporals = [True, False] * 5
        
        self.backbone = nn.Sequential()
        for i in range(len(depthwises)):
            in_channels, out_channels = channels[i], channels[i+1]
            depthwise = depthwises[i]
            temporal = temporals[i]
            
            if temporal:
                self.backbone.append(TemporalBlock(in_channels, out_channels, norms=norms))
            else:
                self.backbone.append(SpatialBlock(in_channels, out_channels, norms=norms))
        
        if detector_head:
            self.head = nn.Sequential(
                TemporalBlock(channels[-1], channels[-1]), 
                nn.Conv3d(channels[-1], channels[-1], (1, 3, 3), (1, 1, 1), (0, 1, 1)), 
                act_layer(), 
                nn.Conv3d(channels[-1], 3, 1), 
            )
        else:
            self.head = nn.Sequential(
                nn.Conv1d(channels[-1], channels[-1], 1), 
                act_layer(), 
                nn.Conv1d(channels[-1], 2, 1), 
            )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.detector:
            return self.head((self.backbone(input)))
        else:
            return self.head(self.backbone(input).mean((-2, -1)))
