"""Standard U-Net (Ronneberger et al., 2015) for SOTA comparison.

Encoder-decoder with skip connections. Chart-generic: outputs only the
charts listed in options['charts'].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class _Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class _Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dh = x2.size(2) - x1.size(2)
        dw = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class UNet(nn.Module):
    """Standard U-Net with 4 down/up stages and skip connections.

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

        f = [64, 128, 256, 512]
        self.inc   = _DoubleConv(in_ch, f[0])
        self.down1 = _Down(f[0], f[1])
        self.down2 = _Down(f[1], f[2])
        self.down3 = _Down(f[2], f[3])
        self.bridge = _Down(f[3], f[3] * 2)
        self.up1   = _Up(f[3] * 2, f[3])
        self.up2   = _Up(f[3],     f[2])
        self.up3   = _Up(f[2],     f[1])
        self.up4   = _Up(f[1],     f[0])
        self.heads = nn.ModuleDict({
            chart: nn.Conv2d(f[0], self.n_classes[chart], 1)
            for chart in self.charts
        })

    def forward(self, x: torch.Tensor) -> dict:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bridge(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return {chart: self.heads[chart](x) for chart in self.charts}
