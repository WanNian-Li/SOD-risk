"""PoolFormer-S24 (Yu et al., 2022) + FPN decoder for SOTA comparison.

MetaFormer backbone using pooling as the token mixer (poolformer_s24),
with a simple FPN head built on top of timm's feature extraction.

Requires: pip install timm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FPNHead(nn.Module):
    """Lightweight FPN head that fuses multi-scale features and produces logits."""

    def __init__(self, feature_channels: list, fpn_channels: int, n_classes: int):
        super().__init__()
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, fpn_channels, 1, bias=False) for c in feature_channels
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True),
            )
            for _ in feature_channels
        ])
        self.cls = nn.Conv2d(fpn_channels, n_classes, 1)

    def forward(self, features: list, output_size: tuple) -> torch.Tensor:
        # Build lateral connections
        laterals = [l(f) for l, f in zip(self.laterals, features)]
        # Top-down path
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode='nearest')
        out = self.fpn_convs[0](laterals[0])
        out = F.interpolate(out, size=output_size, mode='bilinear', align_corners=False)
        return self.cls(out)


class PoolFormerFPN(nn.Module):
    """PoolFormer-S24 backbone + FPN head.

    Uses timm's poolformer_s24 with features_only=True to extract
    multi-scale feature maps, then applies a lightweight FPN decoder.

    PoolFormer-S24 feature channels (4 stages): [64, 128, 320, 512].

    Requires: pip install timm
    """

    def __init__(self, options: dict):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "timm not found. Run: pip install timm"
            ) from e

        in_ch = len(options['train_variables'])
        if options.get('month_encoding', False):
            in_ch += 2
        if options.get('pol_ratio_channel', False):
            in_ch += 1

        self.charts = options['charts']
        self.n_classes = options['n_classes']

        self.backbone = timm.create_model(
            'poolformer_s24',
            pretrained=False,
            features_only=True,
            in_chans=in_ch,
        )
        feature_channels = [fi['num_chs'] for fi in self.backbone.feature_info]

        fpn_ch = 256
        self.heads = nn.ModuleDict({
            chart: _FPNHead(feature_channels, fpn_ch, self.n_classes[chart])
            for chart in self.charts
        })

    def forward(self, x: torch.Tensor) -> dict:
        features = self.backbone(x)
        h, w = x.shape[-2:]
        return {
            chart: self.heads[chart](features, (h, w))
            for chart in self.charts
        }
