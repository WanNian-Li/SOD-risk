"""SegNet (Badrinarayanan et al., 2017) for SOTA comparison.

Encoder-decoder with VGG16-style encoder and max-pooling index based
upsampling. No skip connections; indices are reused from the encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cbr(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SegNet(nn.Module):
    """SegNet with VGG16-style encoder (5 stages) and index-based decoder.

    Input channels are inferred from options['train_variables'] plus any
    optional extra channels (pol_ratio_channel, month_encoding).
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

        # ---------- Encoder ----------
        self.enc1 = nn.Sequential(_cbr(in_ch, 64), _cbr(64, 64))
        self.enc2 = nn.Sequential(_cbr(64, 128),   _cbr(128, 128))
        self.enc3 = nn.Sequential(_cbr(128, 256),  _cbr(256, 256),  _cbr(256, 256))
        self.enc4 = nn.Sequential(_cbr(256, 512),  _cbr(512, 512),  _cbr(512, 512))
        self.enc5 = nn.Sequential(_cbr(512, 512),  _cbr(512, 512),  _cbr(512, 512))

        # ---------- Decoder ----------
        self.dec5 = nn.Sequential(_cbr(512, 512),  _cbr(512, 512),  _cbr(512, 512))
        self.dec4 = nn.Sequential(_cbr(512, 512),  _cbr(512, 512),  _cbr(512, 256))
        self.dec3 = nn.Sequential(_cbr(256, 256),  _cbr(256, 256),  _cbr(256, 128))
        self.dec2 = nn.Sequential(_cbr(128, 128),  _cbr(128, 64))
        self.dec1 = nn.Sequential(_cbr(64, 64))

        self.heads = nn.ModuleDict({
            chart: nn.Conv2d(64, self.n_classes[chart], 1)
            for chart in self.charts
        })

    @staticmethod
    def _pool(x):
        return F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

    @staticmethod
    def _unpool(x, indices, output_size):
        return F.max_unpool2d(x, indices, kernel_size=2, stride=2,
                              output_size=output_size)

    def forward(self, x: torch.Tensor) -> dict:
        # Encode
        s1 = x.shape[-2:]
        x = self.enc1(x)
        x, i1 = self._pool(x)

        s2 = x.shape[-2:]
        x = self.enc2(x)
        x, i2 = self._pool(x)

        s3 = x.shape[-2:]
        x = self.enc3(x)
        x, i3 = self._pool(x)

        s4 = x.shape[-2:]
        x = self.enc4(x)
        x, i4 = self._pool(x)

        s5 = x.shape[-2:]
        x = self.enc5(x)
        x, i5 = self._pool(x)

        # Decode
        x = self._unpool(x, i5, s5)
        x = self.dec5(x)
        x = self._unpool(x, i4, s4)
        x = self.dec4(x)
        x = self._unpool(x, i3, s3)
        x = self.dec3(x)
        x = self._unpool(x, i2, s2)
        x = self.dec2(x)
        x = self._unpool(x, i1, s1)
        x = self.dec1(x)

        return {chart: self.heads[chart](x) for chart in self.charts}
