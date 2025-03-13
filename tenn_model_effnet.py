import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import efficientnet3d_b0

class PatchEmbed3D(nn.Module):
    """Converts 3D input into patch embeddings."""
    def __init__(self, in_channels, embed_dim, patch_size=(2, 4, 4)):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, C, T, H, W) -> (B, D, T//pT, H//pH, W//pW)
        return x

class EfficientNet3DBlock(nn.Module):
    """Replaces ViTBlock with EfficientNet3D."""
    def __init__(self, in_channels):
        super().__init__()
        self.efficient_net = efficientnet3d_b0(pretrained=True)
        self.efficient_net.features[0][0] = nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        return self.efficient_net.features(x)

class SpatialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim=64):
        super().__init__()
        self.patch_embed = PatchEmbed3D(in_channels, embed_dim)
        self.efficient_block = EfficientNet3DBlock(embed_dim)
        self.proj = nn.Conv3d(1280, out_channels, kernel_size=1)  # Mapping EfficientNet output to desired channels

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.efficient_block(x)
        x = self.proj(x)
        return x

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim=64):
        super().__init__()
        self.patch_embed = PatchEmbed3D(in_channels, embed_dim, patch_size=(3, 1, 1))
        self.efficient_block = EfficientNet3DBlock(embed_dim)
        self.proj = nn.Conv3d(1280, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.efficient_block(x)
        x = self.proj(x)
        return x

class TennStEfficientNet(nn.Module):
    def __init__(self, channels, t_kernel_size, n_depthwise_layers, detector_head):
        super().__init__()
        self.detector = detector_head
        depthwises = [False] * (10 - n_depthwise_layers) + [True] * n_depthwise_layers
        temporals = [True, False] * 5

        self.backbone = nn.Sequential()
        for i in range(len(depthwises)):
            in_channels, out_channels = channels[i], channels[i+1]
            temporal = temporals[i]
            
            if temporal:
                self.backbone.append(TemporalBlock(in_channels, out_channels))
            else:
                self.backbone.append(SpatialBlock(in_channels, out_channels))
        
        self.head = nn.Sequential(
            nn.Conv3d(channels[-1], 3, kernel_size=1)
        )
    
    def forward(self, x):
        return self.head(self.backbone(x))
