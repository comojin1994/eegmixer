import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from easydict import EasyDict
from typing import List, Tuple
from models.backbones.factory import FactoryModelHubMixin
from models.backbones.layers import (
    Conv1dWithConstraint,
    Conv2dWithConstraint,
    RMSNorm,
    TransposeLast,
)


def calculate_output_size(input_size, layers):
    output_size = input_size
    for _, _, kernel_size, stride in layers:
        if kernel_size and stride:
            output_size = math.floor((output_size - kernel_size) / stride) + 1
    return output_size


def get_sincos_pos_embed(dim: int, seq_len: int, cls_token: bool = False):
    position = torch.arange(
        seq_len + 1 if cls_token else seq_len, dtype=torch.float
    ).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(position.shape[0], dim)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(position * div_term), torch.cos(
        position * div_term
    )
    return pe.unsqueeze(0)


def apply_rotary_pos_emb(q, k, sin, cos):
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    return (q * cos + rotate_half(q) * sin), (k * cos + rotate_half(k) * sin)


# Building the RoPE cache once, to be shared across layers
class RoPECache(nn.Module):
    def __init__(self, seq_len, head_dim):
        super().__init__()
        sin, cos = self.build_rope_cache(seq_len, head_dim)
        self.register_buffer("sin", sin)
        self.register_buffer("cos", cos)

    @staticmethod
    def build_rope_cache(seq_len, head_dim):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.sin(), emb.cos()


class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.gelu(self.fc(x))


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)


class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts: int = 16):
        super().__init__()
        self.experts = nn.ModuleList(
            [Expert(input_dim, output_dim) for _ in range(num_experts)]
        )
        self.gating_network = GatingNetwork(input_dim, num_experts)

    def forward(self, x):
        gate_weights = self.gating_network(x)
        expert_results = torch.stack([expert(x) for expert in self.experts], dim=-1)
        return torch.sum(expert_results * gate_weights.unsqueeze(2), dim=-1)


class Attention(nn.Module):
    def __init__(
        self, dim, rope_cache, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert head_dim > 0, "head_dim must be positive"

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope_cache = rope_cache

    def forward(self, x):
        b, n, _, h = *x.shape, self.num_heads
        qkv = rearrange(self.qkv(x), "b n (qkv h d) -> qkv b h n d", qkv=3, h=h)
        q, k, v = qkv

        if self.rope_cache:
            sin, cos = self.rope_cache.sin[:n], self.rope_cache.cos[:n]
            q, k = apply_rotary_pos_emb(q, k, sin, cos)

        scale_factor = 1 / math.sqrt(q.size(-1))
        attn_weight = q @ k.transpose(-2, -1) * scale_factor
        attn_weight = F.softmax(attn_weight, dim=-1)

        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)

        out = rearrange(attn, "b h n d -> b n (h d)")
        return self.proj_drop(self.proj(out))


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        rope_cache=None,
        num_experts=16,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        norm_layer=RMSNorm,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            rope_cache=rope_cache,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = norm_layer(dim)
        self.ffn = MixtureOfExperts(
            input_dim=dim, output_dim=dim, num_experts=num_experts
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TemporalSpatialEncoder(nn.Module):

    def __init__(
        self,
        embed_dim,
        nhead,
        num_expert,
        temporal_rope_cache,
        spatial_rope_cache,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        self.temporal_block = Block(
            dim=self.embed_dim,
            num_heads=nhead,
            rope_cache=temporal_rope_cache,
            num_experts=num_expert,
            qkv_bias=False,
            drop=dropout_rate,
            attn_drop=dropout_rate,
        )

        self.spatial_block = Block(
            dim=self.embed_dim,
            num_heads=nhead,
            rope_cache=spatial_rope_cache,
            num_experts=num_expert,
            qkv_bias=False,
            drop=dropout_rate,
            attn_drop=dropout_rate,
        )

        self.infomix_block = nn.Sequential(
            Conv2dWithConstraint(
                self.embed_dim,
                self.embed_dim * 2,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(self.embed_dim * 2),
            nn.GELU(),
            nn.Dropout2d(dropout_rate),
            Conv2dWithConstraint(
                self.embed_dim * 2,
                self.embed_dim,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.Dropout2d(dropout_rate),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, D, T = x.shape

        # Temporal Block
        x = x.reshape(B * C, D, T)  # BC x D x T
        x = x.transpose(1, 2)  # BC x T x D
        x = self.temporal_block(x)
        x = x.reshape(B, C, T, D)  # B x C x T x D
        x = x.transpose(1, 2)  # B x T x C x D

        # Spatial Block
        x = x.reshape(B * T, C, D)  # BT x C x D
        x = self.spatial_block(x)
        x = x.reshape(B, T, C, D)  # B x T x C x D
        x = x.permute(0, 2, 3, 1)  # B x C x D x T

        # InfoMix Block
        x = x.permute(0, 2, 1, 3)  # B x D x C x T
        x = self.infomix_block(x)
        x = x.permute(0, 2, 1, 3)  # B x C x D x T

        return x


class PatchEmbed(nn.Module):
    def __init__(self, layers: List[Tuple[int, int, int, int]], bias: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    Conv1dWithConstraint(in_dim, out_dim, kernel, stride, bias=bias),
                    TransposeLast(),
                    nn.LayerNorm(out_dim),
                    TransposeLast(),
                    nn.GELU(),
                )
                for in_dim, out_dim, kernel, stride in layers
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Embedding(nn.Module):

    def __init__(
        self,
        embed_dim,
        nhead,
        num_experts,
        sinusoidal_dim,
        input_size,
        cnn_layers,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.seq_len = calculate_output_size(input_size, cnn_layers)
        self.sinusoidal_dim = sinusoidal_dim

        self.patch_embed = PatchEmbed(cnn_layers)

        self.spatial_block = Block(
            dim=self.embed_dim,
            num_heads=nhead,
            num_experts=num_experts,
            qkv_bias=False,
            drop=dropout_rate,
            attn_drop=dropout_rate,
        )

        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.sinusoidal_dim + 1, self.embed_dim),
            requires_grad=False,
        )

        self.temporal_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.spatial_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.initialize_weights()

    def initialize_weights(self):
        spatial_pos_embed = get_sincos_pos_embed(
            dim=self.embed_dim,
            seq_len=self.sinusoidal_dim,
            cls_token=True,
        )
        self.spatial_pos_embed.data.copy_(spatial_pos_embed)

        torch.nn.init.normal_(self.spatial_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, C, D, T = x.shape

        # Path embedding
        x = x.reshape(B * C, D, T)  # BC x D x T
        x = self.patch_embed(x)

        # Temporal position embedding & block
        x = x.transpose(1, 2)  # BC x T x D
        token = self.temporal_token
        token = token.expand(B * C, -1, -1)
        x = torch.cat((token, x), dim=1)
        x = x.reshape(B, C, -1, self.embed_dim)  # B x C x T x D
        x = x.transpose(1, 2)  # B x T x C x D

        # Spatial position embedding & block
        B, T, C, D = x.shape
        x = x.reshape(B * T, C, D)  # BT x C x D
        token = self.spatial_token.expand(B * T, -1, -1)
        x = torch.cat((token, x), dim=1)
        x = x + self.spatial_pos_embed[:, : C + 1, :]
        x = self.spatial_block(x)
        x = x.reshape(B, -1, C + 1, self.embed_dim)  # B x T x C x D
        x = x.permute(0, 2, 3, 1)  # B x C x D x T

        return x


class ClassifierHead(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_classes,
        seq_len,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self._init_mlp_head(embed_dim, num_classes, seq_len, dropout_rate)

    def _init_mlp_head(self, embed_dim, num_classes, seq_len, dropout_rate):

        self.gate_module = Block(
            dim=self.embed_dim,
            num_heads=4,
            num_experts=1,
            qkv_bias=False,
            drop=dropout_rate,
            attn_drop=dropout_rate,
        )

        self.norm = nn.LayerNorm(self.embed_dim)

        self.mlp_head = nn.Sequential(
            Conv1dWithConstraint(
                embed_dim,
                embed_dim,
                kernel_size=(seq_len + 1) // 4,
                padding="same",
                groups=embed_dim,
                bias=False,
            ),
            Conv1dWithConstraint(
                embed_dim,
                embed_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=(seq_len + 1) // 8, stride=(seq_len + 1) // 8),
            nn.Dropout1d(dropout_rate),
            nn.Flatten(),
            nn.Linear(8 * embed_dim, num_classes),
        )

    def forward(self, x, debug=False):

        x = x.permute(0, 2, 1, 3)  # B x D x C x T

        s = x.permute(0, 3, 2, 1)
        B, T, C, D = s.shape
        s = s.reshape(B * T, C, D)
        gate_weights = F.softmax(self.gate_module(s), dim=1)  # BT x C x D
        gate_weights = gate_weights.reshape(B, T, C, D)  # B x T x C x D
        gate_weights = gate_weights.permute(0, 3, 2, 1)  # B x D x C x T

        x = torch.mean(x * gate_weights, dim=2)  # B x D x T
        x = x.transpose(1, 2)  # B x T x D
        x = self.norm(x)
        x = x.transpose(1, 2)  # B x D x T

        x = self.mlp_head(x)

        if debug:
            return x, gate_weights
        else:
            return x


class EEGMixer(nn.Module, FactoryModelHubMixin):
    def __init__(self, args: EasyDict):
        super().__init__()

        self.embed_dim = args.dim

        # Initialize the shared RoPE cache once
        temporal_rope_cache = RoPECache(
            args.temporal_rope_dim, self.embed_dim // args.nhead
        )
        spatial_rope_cache = RoPECache(
            args.spatial_rope_dim, self.embed_dim // args.nhead
        )

        self.embedding = Embedding(
            self.embed_dim,
            args.nhead,
            args.num_experts,
            args.sinusoidal_dim,
            args.origin_ival[-1],
            args.cnn_layers,
            args.dropout_rate,
        )

        self.blocks = nn.ModuleList(
            [
                TemporalSpatialEncoder(
                    self.embed_dim,
                    args.nhead,
                    args.num_experts,
                    temporal_rope_cache,
                    spatial_rope_cache,
                    args.dropout_rate,
                )
                for _ in range(args.nlayer)
            ]
        )

        self.classifier_head = ClassifierHead(
            self.embed_dim,
            args.num_classes,
            self.embedding.seq_len,
            args.dropout_rate,
        )

    def forward(self, x, debug=False):
        x = x.permute(0, 2, 1, 3)  # B x C x D x T
        x = self.embedding(x)

        for block in self.blocks:
            x = x + block(x)  # B x C x D x T

        x = self.classifier_head(x, debug=debug)

        return x
