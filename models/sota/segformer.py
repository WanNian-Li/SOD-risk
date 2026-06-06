"""SegFormer-B2 (Xie et al., 2021) — no mmseg required.

Uses SMP's MiT-B2 encoder backbone for feature extraction plus a
pure-PyTorch all-MLP SegformerHead decoder.
Only requires segmentation_models_pytorch (already a project dependency).

MiT-B2 stage channels: [64, 128, 320, 512] at strides [4, 8, 16, 32].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SegformerHead(nn.Module):
    """All-MLP decoder: project each scale to embed_dim, upsample to
    finest scale, concatenate, fuse, dropout, classify."""

    def __init__(self, in_channels: list, embed_dim: int, n_classes: int):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Conv2d(c, embed_dim, 1, bias=False) for c in in_channels
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(0.1)
        self.cls = nn.Conv2d(embed_dim, n_classes, 1)

    def forward(self, features: list) -> torch.Tensor:
        target = features[0].shape[-2:]
        projected = []
        for feat, proj in zip(features, self.projections):
            x = proj(feat)
            if x.shape[-2:] != target:
                x = F.interpolate(x, size=target, mode='bilinear', align_corners=False)
            projected.append(x)
        x = torch.cat(projected, dim=1)
        x = self.fuse(x)
        x = self.dropout(x)
        return self.cls(x)


class SegFormer(nn.Module):
    """SegFormer-B2: MiT-B2 encoder (via SMP) + SegformerHead.

    Requires: pip install segmentation-models-pytorch
    """

    def __init__(self, options: dict):
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except ImportError as e:
            raise ImportError(
                "segmentation_models_pytorch not found. "
                "Run: pip install segmentation-models-pytorch"
            ) from e

        in_ch = len(options['train_variables'])
        if options.get('month_encoding', False):
            in_ch += 2
        if options.get('pol_ratio_channel', False):
            in_ch += 1

        self.charts = options['charts']
        self.n_classes = options['n_classes']

        # Build a temporary SMP FPN model to extract the MiT-B2 encoder.
        # The encoder handles the full ViT backbone; we replace the FPN decoder
        # with our own SegformerHead.
        _tmp = smp.FPN(
            encoder_name='mit_b2',
            encoder_weights=None,
            in_channels=in_ch,
            classes=1,
        )
        self.encoder = _tmp.encoder
        del _tmp

        # MiT-B2 outputs 4 transformer stages with these channel sizes.
        # SMP encoder.out_channels typically looks like (in_ch, 64, 128, 320, 512).
        # We take the last 4 entries (all transformer stages).
        feat_channels = list(self.encoder.out_channels[-4:])

        embed_dim = 256
        self.heads = nn.ModuleDict({
            chart: _SegformerHead(feat_channels, embed_dim, self.n_classes[chart])
            for chart in self.charts
        })

    def forward(self, x: torch.Tensor) -> dict:
        all_feats = self.encoder(x)   # tuple of feature maps
        # Use the last 4 features (MiT-B2 stages 1–4)
        features = list(all_feats[-4:])
        out = {}
        for chart in self.charts:
            logits = self.heads[chart](features)
            if logits.shape[-2:] != x.shape[-2:]:
                logits = F.interpolate(logits, size=x.shape[-2:],
                                       mode='bilinear', align_corners=False)
            out[chart] = logits
        return out
