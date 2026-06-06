"""ResNet50 + FPN decoder for SOTA comparison.

Uses segmentation_models_pytorch (SMP) with a ResNet50 encoder and FPN
(Feature Pyramid Network) decoder. One independent head per output chart.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetFPN(nn.Module):
    """ResNet50 encoder + FPN decoder (SMP).

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

        self.models = nn.ModuleDict({
            chart: smp.FPN(
                encoder_name='resnet50',
                encoder_weights=None,
                in_channels=in_ch,
                classes=self.n_classes[chart],
            )
            for chart in self.charts
        })

    def forward(self, x: torch.Tensor) -> dict:
        out = {}
        for chart in self.charts:
            logits = self.models[chart](x)
            if logits.shape[-2:] != x.shape[-2:]:
                logits = F.interpolate(logits, size=x.shape[-2:],
                                       mode='bilinear', align_corners=False)
            out[chart] = logits
        return out
