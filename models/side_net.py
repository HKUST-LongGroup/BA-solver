### The SideNet model implementation. Done###

import torch
import torch.nn as nn
import math

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.positional_embedding(t, self.frequency_embedding_size).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < -1.0 #self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class ResBlock(nn.Module):
    def __init__(self, channels, emb_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act = nn.SiLU()
        self.emb_proj = nn.Linear(emb_channels, channels * 2)

    def forward(self, x, emb):
        h = self.norm1(x)
        h = self.conv1(h)
        style = self.emb_proj(emb).unsqueeze(-1).unsqueeze(-1)
        scale, shift = style.chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.act(h)
        h = self.conv2(h)
        return x + h

class GlobalSelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x_in = x.permute(0, 2, 3, 1).view(B, H * W, C)
        x_norm = self.norm(x).permute(0, 2, 3, 1).view(B, H * W, C)
        
        qkv = self.qkv(x_norm).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x_out = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        x_out = self.proj(x_out)
        
        return (x_in + x_out).view(B, H, W, C).permute(0, 3, 1, 2)

class BASideNet(nn.Module):
    def __init__(self, in_channels=4, base_channels=64, h_emb_dim=256, num_layers=3, num_classes=1000):
        super().__init__()
        
        self.h_embedder = TimestepEmbedder(h_emb_dim)
        self.t_embedder = TimestepEmbedder(h_emb_dim)
        self.y_embedder = LabelEmbedder(num_classes, h_emb_dim, dropout_prob=0.1)
        
        self.emb_fusion = nn.Sequential(
            nn.Linear(h_emb_dim * 3, h_emb_dim),
            nn.SiLU(),
            nn.Linear(h_emb_dim, h_emb_dim)
        )

        self.input_conv = nn.Conv2d(in_channels * 2, base_channels, 3, 1, 1)
        
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(ResBlock(base_channels, h_emb_dim))
            if i == num_layers // 2: 
                self.blocks.append(GlobalSelfAttention(base_channels))
        
        self.output_norm = nn.GroupNorm(8, base_channels)
        self.output_conv = nn.Conv2d(base_channels, in_channels, 3, 1, 1)
        
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, x, v_sit, t, h, y):
        # 1. Embeddings
        t_emb = self.t_embedder(t)
        h_emb = self.h_embedder(h)
        
        if y is not None:
            # The outer Loss/Sampler has already processed Dropout.
            # So we do not need to apply Dropout again.
            y_emb = self.y_embedder(y, train=False)
        else:
            y_emb = torch.zeros_like(t_emb)
        
        raw_emb = torch.cat([t_emb, h_emb, y_emb], dim=-1)
        emb = self.emb_fusion(raw_emb)
        
        # 2. Network Body
        net_in = torch.cat([x, v_sit], dim=1)
        feat = self.input_conv(net_in)
        
        for block in self.blocks:
            if isinstance(block, ResBlock):
                feat = block(feat, emb)
            else:
                feat = block(feat)
            
        feat = self.output_norm(feat)
        feat = torch.nn.functional.silu(feat)
        delta_v = self.output_conv(feat)
        
        # 3. Hard Constraint
        h_scale = h.view(-1, 1, 1, 1)
        delta_v = delta_v * h_scale
        
        return delta_v

