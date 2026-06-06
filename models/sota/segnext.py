"""SegNeXt-T (Guo et al., 2022) — pure PyTorch, no mmseg required.

Full MSCAN-T (Multi-Scale Convolutional Attention Network, Tiny) backbone
implemented from scratch, with a simplified aggregation head.

MSCAN-T config:
  embed_dims  = [32, 64, 160, 256]
  depths      = [3, 3, 5, 2]
  mlp_ratios  = [8, 8, 4, 4]
  strides     = [4, 2, 2, 2]   (overlap patch embeddings)

Head: projects stages 2-4 (channels [64, 160, 256]) to 256 ch,
upsamples to stride-8 resolution, fuses and classifies.

Only requires standard PyTorch — no extra dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

class _DropPath(nn.Module):
    """Stochastic depth regularisation (per sample)."""
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(
            torch.full(shape, keep, dtype=x.dtype, device=x.device))
        return x * mask / keep


class _OPE(nn.Module):
    """Overlapping Patch Embedding: 3×3 conv, stride ≥ 2, BN."""
    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                              padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


# --------------------------------------------------------------------------- #
# MSCA (Multi-Scale Convolutional Attention)
# --------------------------------------------------------------------------- #

class _MSCA(nn.Module):
    """
    Multi-Scale Convolutional Attention module.

    1. Depth-wise 5×5 conv + BN
    2. Four attention branches:
         - point-wise 1×1
         - 1×7 + 7×1 depth-wise strip convs
         - 1×11 + 11×1 depth-wise strip convs
         - 1×21 + 21×1 depth-wise strip convs
    3. Sum branches, 1×1 projection
    4. Multiply (gate) with original input
    """

    def __init__(self, channels: int):
        super().__init__()
        self.dw5  = nn.Conv2d(channels, channels, 5, padding=2,
                              groups=channels, bias=False)
        self.bn   = nn.BatchNorm2d(channels)

        self.pw0  = nn.Conv2d(channels, channels, 1, bias=False)
        self.dw1  = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 7),  padding=(0, 3),  groups=channels, bias=False),
            nn.Conv2d(channels, channels, (7, 1),  padding=(3, 0),  groups=channels, bias=False),
        )
        self.dw2  = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 11), padding=(0, 5),  groups=channels, bias=False),
            nn.Conv2d(channels, channels, (11, 1), padding=(5, 0),  groups=channels, bias=False),
        )
        self.dw3  = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 21), padding=(0, 10), groups=channels, bias=False),
            nn.Conv2d(channels, channels, (21, 1), padding=(10, 0), groups=channels, bias=False),
        )
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x
        x = self.bn(self.dw5(x))
        attn = self.pw0(x) + self.dw1(x) + self.dw2(x) + self.dw3(x)
        return u * self.proj(attn)


class _FFN(nn.Module):
    """Point-wise feed-forward network: expand → GELU → project."""
    def __init__(self, channels: int, ratio: int = 4):
        super().__init__()
        hidden = channels * ratio
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _MSCANBlock(nn.Module):
    """One MSCAN block: MSCA + FFN, each with BN pre-norm and residual."""
    def __init__(self, channels: int, mlp_ratio: int = 4,
                 drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.attn  = _MSCA(channels)
        self.norm2 = nn.BatchNorm2d(channels)
        self.ffn   = _FFN(channels, mlp_ratio)
        self.dp    = _DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dp(self.attn(self.norm1(x)))
        x = x + self.dp(self.ffn(self.norm2(x)))
        return x


# --------------------------------------------------------------------------- #
# MSCAN-T backbone
# --------------------------------------------------------------------------- #

class _MSCAN_T(nn.Module):
    """
    MSCAN-T backbone — outputs 4 feature maps at strides [4, 8, 16, 32].
    Channel dims: [32, 64, 160, 256].
    """
    _DIMS    = [32, 64, 160, 256]
    _DEPTHS  = [3,  3,  5,   2]
    _RATIOS  = [8,  8,  4,   4]
    _STRIDES = [4,  2,  2,   2]

    def __init__(self, in_ch: int, drop_path_rate: float = 0.1):
        super().__init__()
        total_blocks = sum(self._DEPTHS)
        dp_rates = [x.item() for x in
                    torch.linspace(0, drop_path_rate, total_blocks)]

        self.embeds = nn.ModuleList()
        self.stages = nn.ModuleList()
        idx = 0
        prev_ch = in_ch
        for d, depth in enumerate(self._DEPTHS):
            self.embeds.append(_OPE(prev_ch, self._DIMS[d], self._STRIDES[d]))
            blocks = nn.Sequential(*[
                _MSCANBlock(self._DIMS[d], self._RATIOS[d], dp_rates[idx + j])
                for j in range(depth)
            ])
            self.stages.append(blocks)
            idx += depth
            prev_ch = self._DIMS[d]

    def forward(self, x: torch.Tensor):
        feats = []
        for embed, stage in zip(self.embeds, self.stages):
            x = stage(embed(x))
            feats.append(x)
        return feats   # [s1(32ch), s2(64ch), s3(160ch), s4(256ch)]


# --------------------------------------------------------------------------- #
# Aggregation head (simplified LightHamHead)
# --------------------------------------------------------------------------- #

class _AggHead(nn.Module):
    """
    Lightweight aggregation head: project each stage to head_ch,
    upsample all to the finest stage resolution, fuse, classify.
    """

    def __init__(self, in_channels: list, head_ch: int, n_classes: int):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, head_ch, 1, bias=False),
                nn.BatchNorm2d(head_ch),
                nn.ReLU(inplace=True),
            )
            for c in in_channels
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(head_ch * len(in_channels), head_ch, 1, bias=False),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(0.1)
        self.cls = nn.Conv2d(head_ch, n_classes, 1)

    def forward(self, feats: list, out_size: tuple) -> torch.Tensor:
        ref = feats[0].shape[-2:]
        projected = []
        for f, p in zip(feats, self.projs):
            x = p(f)
            if x.shape[-2:] != ref:
                x = F.interpolate(x, size=ref, mode='bilinear', align_corners=False)
            projected.append(x)
        x = self.fuse(torch.cat(projected, dim=1))
        x = self.dropout(x)
        x = self.cls(x)
        if x.shape[-2:] != out_size:
            x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
        return x


# --------------------------------------------------------------------------- #
# SegNeXt
# --------------------------------------------------------------------------- #

class SegNeXt(nn.Module):
    """
    SegNeXt-T: MSCAN-T backbone + aggregation head.
    Pure PyTorch — no mmseg required.

    Head uses stages 2, 3, 4 (channels [64, 160, 256]).
    """

    def __init__(self, options: dict):
        super().__init__()
        in_ch = len(options['train_variables'])
        if options.get('month_encoding', False):
            in_ch += 2
        if options.get('pol_ratio_channel', False):
            in_ch += 1

        self.charts = options['charts']
        self.n_classes = options['n_classes']

        self.backbone = _MSCAN_T(in_ch=in_ch, drop_path_rate=0.1)

        # Use the last 3 stages: channels [64, 160, 256]
        head_in_ch = [64, 160, 256]
        self.heads = nn.ModuleDict({
            chart: _AggHead(head_in_ch, 256, self.n_classes[chart])
            for chart in self.charts
        })

    def forward(self, x: torch.Tensor) -> dict:
        _, s2, s3, s4 = self.backbone(x)   # discard s1 (32ch, coarsest feature)
        feats = [s2, s3, s4]
        h, w = x.shape[-2:]
        return {chart: self.heads[chart](feats, (h, w)) for chart in self.charts}
