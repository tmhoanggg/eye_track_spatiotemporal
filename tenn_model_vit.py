import torch
import torch.nn as nn
from torch.nn import functional as F

import warnings

warnings.formatwarning = lambda message, category, filename, lineno, line=None: \
    f'{category.__name__}: {message}\n'


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.embed_dim = embed_dim

    def forward(self, x):
        # Input: (B*T, C, H, W)
        x = self.proj(x)  # (B*T, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B*T, num_patches, embed_dim)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input: (num_patches, B*T, embed_dim)
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_out))
        return x

class ViT(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_heads, ff_dim, num_layers, out_channels):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, (224//patch_size)**2, embed_dim))  # Giả định H=W=224
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.to_out = nn.Linear(embed_dim, out_channels)

    def forward(self, x):
        # Input: (B*T, C, H, W)
        B_T, C, H, W = x.shape
        x = self.patch_embed(x)  # (B*T, num_patches, embed_dim)
        x = x + self.pos_embed  # Add positional embedding
        x = x.transpose(0, 1)  # (num_patches, B*T, embed_dim)
        for block in self.transformer:
            x = block(x)
        x = x.transpose(0, 1)  # (B*T, num_patches, embed_dim)
        x = x.mean(dim=1)  # Global average pooling: (B*T, embed_dim)
        x = self.to_out(x)  # (B*T, out_channels)
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
pw_conv = lambda in_channels, out_channels: nn.Conv3d(in_channels, out_channels, 1, bias=False)


class SpatialBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 depthwise=False,  # Không cần nữa vì dùng ViT
                 kernel_size=1,    # Không cần nữa
                 full_conv3d=False,
                 norms='mixed'):
        super().__init__()
        self.norms = norms
        self.streaming_mode = False
        self.fifo = None  # for streaming inference

        # Thay vì dùng Conv3d, dùng ViT
        patch_size = 16  # Ví dụ: patch size 16x16
        embed_dim = 256  # Kích thước embedding
        num_heads = 8    # Số head trong self-attention
        ff_dim = 512     # Kích thước feed-forward
        num_layers = 4   # Số lớp transformer

        self.vit = ViT(in_channels, patch_size, embed_dim, num_heads, ff_dim, num_layers, out_channels)

        # Norm block sau ViT
        if self.norms == 'all_gn':
            self.norm = gn_block(out_channels)
        else:
            self.norm = bn_block(out_channels)

    def streaming(self, enabled=True):
        if enabled:
            assert not self.training, "Can only use streaming mode during evaluation."
        self.streaming_mode = enabled
        
    def reset_memory(self):
        self.fifo = None
    
    def forward(self, input):
        # Input: (B, C, T, H, W)
        B, C, T, H, W = input.shape
        x = input.moveaxis(2, 1).reshape(B*T, C, H, W)  # (B*T, C, H, W)
        x = self.vit(x)  # (B*T, out_channels)
        x = x.reshape(B, T, -1).moveaxis(1, 2)  # (B, out_channels, T)
        x = x.unsqueeze(-1).unsqueeze(-1)  # (B, out_channels, T, 1, 1)
        return self.norm(x)


class TemporalBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3, 
                 depthwise=False,
                 full_conv3d=False, 
                 norms='mixed'):
        super().__init__()
        assert out_channels % 4 == 0  # needed for group norm to work
        self.kernel_size = kernel_size
        self.depthwise = depthwise
        self.norms = norms
        kernel = (kernel_size,3,3) if full_conv3d else (kernel_size,1,1)
        
        self.streaming_mode = False
        self.fifo = None  # for streaming inference
        
        if self.norms=='mixed':
            norm1_block = bn_block
            norm2_block = gn_block
        elif self.norms=='all_bn':
            norm1_block = bn_block
            norm2_block = bn_block
        elif self.norms=='all_gn':
            norm1_block = gn_block
            norm2_block = gn_block

        if depthwise:
            self.block = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel, groups=in_channels, bias=False), 
                norm1_block(in_channels), 
                pw_conv(in_channels, out_channels), 
                norm2_block(out_channels), 
            )
            
        else:
            self.block = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel, bias=False), 
                norm2_block(out_channels), 
            )

    def streaming(self, enabled=True):
        if enabled:
            assert not self.training, "Can only use streaming mode during evaluation."
        self.streaming_mode = enabled
        
    def reset_memory(self):
        self.fifo = None
    
    def forward(self, input):
        if self.streaming_mode:
            return self._streaming_forward(input)
                  
        input = F.pad(input, (0, 0, 0, 0, self.kernel_size - 1, 0))
        return self.block(input)
    
    def _streaming_forward(self, input):
        if self.fifo is None:
            self.fifo = torch.zeros(*input.shape[:2], self.kernel_size, *input.shape[3:]).type_as(input)
        self.fifo = torch.cat([self.fifo[:, :, 1:], input], dim=2)
        return self.block(self.fifo)
        

class TennStViT(nn.Module):
    def __init__(
        self, 
        channels, 
        t_kernel_size, 
        n_depthwise_layers, 
        detector_head, 
        detector_depthwise, 
        full_conv3d=False,
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
                self.backbone.append(TemporalBlock(in_channels, out_channels, 
                                                   kernel_size=t_kernel_size, depthwise=depthwise,
                                                   full_conv3d=full_conv3d, norms=norms))
            else:
                self.backbone.append(SpatialBlock(in_channels, out_channels, depthwise=depthwise,
                                                  full_conv3d=full_conv3d,
                                                  kernel_size=t_kernel_size if full_conv3d else 1,
                                                  norms=norms))
        
        if detector_head:
            self.head = nn.Sequential(
                TemporalBlock(channels[-1], channels[-1], t_kernel_size, depthwise=detector_depthwise), 
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
    
    def streaming(self, enabled=True):
        if enabled:
            warnings.warn("You have enabled the streaming mode of the network. It is expected, but not checked, that the input will be of shape (batch, 1, H, W).")
        for name, module in self.named_modules():
            if name and hasattr(module, 'streaming'):
                module.streaming(enabled)
                
    def reset_memory(self):
        for name, module in self.named_modules():
            if name and hasattr(module, 'reset_memory'):
                module.reset_memory()
         
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.detector:
            return self.head((self.backbone(input)))
        else:
            return self.head(self.backbone(input).mean((-2, -1)))