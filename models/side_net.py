import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FastTimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, max_period=10000.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(inplace=True), 
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

        half = frequency_embedding_size // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t):
        args = t.view(-1, 1).float() * self.freqs.view(1, -1)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb.to(dtype=t.dtype))

class FastLabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)

    def forward(self, labels):
        return self.embedding_table(labels)

class FastChannelRMSNorm(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + 1e-5)

class NanoResBlock(nn.Module):
    def __init__(self, channels, emb_channels):
        super().__init__()
        self.norm1 = FastChannelRMSNorm()
        
        self.conv1_dw = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.conv1_pw = nn.Conv2d(channels, channels, 1, 1, 0, bias=False)
        
        self.norm2 = FastChannelRMSNorm()
        self.conv2_dw = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.conv2_pw = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        
        self.act = nn.SiLU(inplace=True)
        self.emb_proj = nn.Linear(emb_channels, channels * 2)

    def forward(self, x, emb):
        h = self.conv1_pw(self.conv1_dw(self.norm1(x)))
        
        style = self.emb_proj(emb).view(-1, x.shape[1] * 2, 1, 1)
        scale, shift = style.chunk(2, dim=1)
        
        h = self.norm2(h)
        h = torch.addcmul(shift, h, scale + 1.0)
        
        h = self.conv2_pw(self.conv2_dw(self.act(h)))
        return x + h

class FastFlashAttention(nn.Module):
    def __init__(self, channels, emb_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.norm = FastChannelRMSNorm()
        self.emb_proj = nn.Linear(emb_channels, channels * 2)

        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x, emb):
        B, C, H, W = x.shape
        N = H * W

        h = self.norm(x)
        style = self.emb_proj(emb).view(-1, C * 2, 1, 1)
        scale, shift = style.chunk(2, dim=1)
        h = torch.addcmul(shift, h, scale + 1.0)

        qkv = self.qkv(h) # (B, 3C, H, W)
        qkv = qkv.view(B, 3, self.num_heads, self.head_dim, N).permute(1, 0, 2, 4, 3)
        q, k, v = qkv.unbind(dim=0)

        attn_out = F.scaled_dot_product_attention(q, k, v)

        attn_out = attn_out.transpose(2, 3).reshape(B, C, H, W)
        out = self.proj(attn_out)

        return x + out

class BASideNet(nn.Module):
    def __init__(self, in_channels=4, base_channels=64, h_emb_dim=256, num_layers=4, num_classes=1000):
        super().__init__()
        
        self.h_embedder = FastTimestepEmbedder(h_emb_dim)
        self.t_embedder = FastTimestepEmbedder(h_emb_dim)
        self.y_embedder = FastLabelEmbedder(num_classes, h_emb_dim)
        
        self.emb_fusion = nn.Sequential(
            nn.Linear(h_emb_dim * 3, h_emb_dim),
            nn.SiLU(inplace=True),
            nn.Linear(h_emb_dim, h_emb_dim)
        )

        self.input_conv = nn.Conv2d(in_channels * 2, base_channels, 3, 1, 1, bias=False)
        
        self.blocks = nn.ModuleList()
        mid_point = num_layers // 2
        for i in range(num_layers):
            if i == mid_point:
                self.blocks.append(FastFlashAttention(base_channels, h_emb_dim))
            else:
                self.blocks.append(NanoResBlock(base_channels, h_emb_dim))
        
        self.output_norm = FastChannelRMSNorm()
        self.output_conv = nn.Conv2d(base_channels, in_channels, 3, 1, 1)
        
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, x, v_sit, t, h, y):
        t_emb = self.t_embedder(t)
        h_emb = self.h_embedder(h)
        y_emb = self.y_embedder(y)
        
        emb = self.emb_fusion(torch.cat([t_emb, h_emb, y_emb], dim=-1))
        
        feat = self.input_conv(torch.cat([x, v_sit], dim=1))
        
        for block in self.blocks:
            feat = block(feat, emb)
            
        feat = F.silu(self.output_norm(feat), inplace=True)
        delta_v = self.output_conv(feat)
        
        return delta_v * h.view(-1, 1, 1, 1)
