# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):

        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 use_adaptive_gate=True, use_pol_bias=True):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.use_adaptive_gate = use_adaptive_gate
        self.use_pol_bias = use_pol_bias

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        # Fixed scalar logit gate (baseline RealFormer style, used when use_adaptive_gate=False).
        self.alpha = nn.Parameter(torch.zeros(num_heads))
        # Content-adaptive logit gate (micro layer, used when use_adaptive_gate=True).
        # Weights are re-zeroed by SwinTransformer._reset_pa2_weights() after
        # apply(_init_weights), so sigmoid(0)=0.5 at init → neutral residual blend.
        self.alpha_gate = nn.Sequential(
            nn.Linear(dim, num_heads),
            nn.Sigmoid()
        )
        # Polarization prior bias projection (physical layer).
        # Zero-init → no pol bias at training start; _reset_pa2_weights() ensures this.
        self.pol_proj = nn.Linear(1, num_heads, bias=False)

    def forward(self, x, mask=None, attn_residual=None, pol_feat=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
            attn_residual: pre-softmax attention logits from the previous same-type
                           block, shape (num_windows*B, num_heads, N, N) or None.
            pol_feat: per-token polarization ratio, shape (num_windows*B, N, 1) or None.
        Returns:
            x: output features, shape (num_windows*B, N, C)
            pre_softmax_attn: attention logits before softmax, to be passed as
                              attn_residual to the next same-type block.
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        # SW-MSA cyclic-shift mask must come before pol_bias and attn_residual so
        # that masked positions remain suppressed even after the additive terms.
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        # ① Polarization prior bias B_pol (physical layer).
        # pol_feat: (B_, N, 1); pol_sim: (B_, N, N); pol_bias: (B_, nH, N, N)
        if pol_feat is not None and self.use_pol_bias:
            pol_sim  = pol_feat @ pol_feat.transpose(-2, -1)       # (B_, N, N)
            pol_bias = self.pol_proj(pol_sim.unsqueeze(-1))         # (B_, N, N, nH)
            pol_bias = pol_bias.permute(0, 3, 1, 2)                # (B_, nH, N, N)
            attn = attn + pol_bias

        # ② Logit residual gate (micro layer).
        # use_adaptive_gate=True  → content-adaptive per-window gate (PA² contribution)
        # use_adaptive_gate=False → fixed scalar alpha (baseline RealFormer style)
        if attn_residual is not None:
            if self.use_adaptive_gate:
                gate = self.alpha_gate(x.mean(dim=1))              # (B_, num_heads)
                gate = gate.view(B_, self.num_heads, 1, 1)
                attn = attn + gate * attn_residual
            else:
                alpha = self.alpha.view(1, self.num_heads, 1, 1)
                attn = attn + alpha * attn_residual

        pre_softmax_attn = attn  # saved for the next same-type block (no detach: full grad flow)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, pre_softmax_attn

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_kimi_attnres=False,
                 use_adaptive_gate=True, use_pol_bias=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_kimi_attnres = use_kimi_attnres
        self.use_pol_bias = use_pol_bias
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            use_adaptive_gate=use_adaptive_gate, use_pol_bias=use_pol_bias)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        # Kimi Block AttnRes parameters (macro layer, Stage 2 only)
        if use_kimi_attnres:
            self.w_attn   = nn.Parameter(torch.zeros(dim))  # pseudo-query for attn sub-layer
            self.w_mlp    = nn.Parameter(torch.zeros(dim))  # pseudo-query for MLP sub-layer
            self.kimi_norm = norm_layer(dim)                 # prevents magnitude dominance in retrieval

    def forward(self, x, attn_residual=None, history_blocks=None, pol_map=None):
        """
        Args:
            x: input token sequence, shape (B, H*W, C)
            attn_residual: pre-softmax logits from the previous same-type block,
                           shape (nW*B, num_heads, N, N) or None.
            history_blocks: list of prior block outputs (B, H*W, C), managed by
                            BasicLayer. None for non-Stage-2 blocks.
            pol_map: polarization ratio map, shape (B, H*W, 1) or None.
        Returns:
            partial_block: output token sequence, shape (B, H*W, C)
            pre_softmax_attn: attention logits before softmax (nW*B, num_heads, N, N)
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # ── Track 1: residual baseline (gradient highway, never replaced) ──
        partial_block = x

        # ── Attn sub-layer: retrieve input h from history (Kimi macro layer) ──
        if self.use_kimi_attnres and history_blocks:
            h = self._kimi_retrieve(history_blocks, partial_block, self.w_attn)
        else:
            h = partial_block

        # Norm → 2-D → cyclic shift → window partition  (all on h, not partial_block)
        h_norm   = self.norm1(h)
        h_2d     = h_norm.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_h = torch.roll(h_2d, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_h = h_2d
        x_windows = window_partition(shifted_h, self.window_size).view(-1, self.window_size ** 2, C)

        # pol_map window partition — only when use_pol_bias is active
        pol_windows = None
        if pol_map is not None and self.use_pol_bias:
            pol_2d = pol_map.view(B, H, W, 1)
            if self.shift_size > 0:
                pol_2d = torch.roll(pol_2d, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            pol_windows = window_partition(pol_2d, self.window_size).view(-1, self.window_size ** 2, 1)

        attn_out_windows, pre_softmax_attn = self.attn(
            x_windows, mask=self.attn_mask, attn_residual=attn_residual, pol_feat=pol_windows
        )

        # Merge windows → reverse cyclic shift → flatten
        attn_out_windows = attn_out_windows.view(-1, self.window_size, self.window_size, C)
        shifted_out      = window_reverse(attn_out_windows, self.window_size, H, W)
        if self.shift_size > 0:
            attn_out = torch.roll(shifted_out, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_out = shifted_out
        attn_out = attn_out.view(B, L, C)

        # ── Track 1 update (attn sub-layer) ──
        partial_block = partial_block + self.drop_path(attn_out)

        # ── MLP sub-layer: retrieve updated h from history ──
        if self.use_kimi_attnres and history_blocks:
            h = self._kimi_retrieve(history_blocks, partial_block, self.w_mlp)
        else:
            h = partial_block

        mlp_out = self.mlp(self.norm2(h))

        # ── Track 1 update (MLP sub-layer) ──
        partial_block = partial_block + self.drop_path(mlp_out)

        return partial_block, pre_softmax_attn

    def _kimi_retrieve(self, history_blocks, partial_block, pseudo_query):
        """Retrieve sub-layer input h from history + partial_block via per-token softmax attention.

        V shape: (N+1, B, L, C) where N+1 = len(history)+1 (partial_block appended last).
        Returns h: (B, L, C) — the weighted combination used as the sub-layer input.
        """
        V      = torch.stack(history_blocks + [partial_block], dim=0)   # (N+1, B, L, C)
        K      = self.kimi_norm(V)                                        # (N+1, B, L, C)
        logits = torch.einsum('c, n b l c -> n b l', pseudo_query, K)    # (N+1, B, L)
        alpha  = logits.softmax(dim=0)                                    # (N+1, B, L)
        return torch.einsum('n b l, n b l c -> b l c', alpha, V)          # (B, L, C)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 use_kimi_attnres=False, use_adaptive_gate=True, use_pol_bias=True):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.use_kimi_attnres = use_kimi_attnres

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 use_kimi_attnres=use_kimi_attnres,
                                 use_adaptive_gate=use_adaptive_gate,
                                 use_pol_bias=use_pol_bias)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, pol_map=None):
        attn_res_wmsa  = None
        attn_res_swmsa = None
        # history_blocks accumulates block outputs for Kimi AttnRes (Stage 2 only).
        # An empty list is falsy, so Block 0 degrades to standard Swin on first pass.
        history_blocks = [] if self.use_kimi_attnres else None

        for i, blk in enumerate(self.blocks):
            current_residual = attn_res_wmsa if i % 2 == 0 else attn_res_swmsa

            if self.use_checkpoint:
                result = checkpoint.checkpoint(blk, x)
            else:
                result = blk(x,
                             attn_residual=current_residual,
                             history_blocks=history_blocks,
                             pol_map=pol_map)

            x, pre_softmax_attn = result

            if i % 2 == 0:
                attn_res_wmsa = pre_softmax_attn
            else:
                attn_res_swmsa = pre_softmax_attn

            # Append completed block output to history (no detach: full gradient highway)
            if self.use_kimi_attnres:
                history_blocks.append(x)

        if self.downsample is not None:
            x = self.downsample(x)
            if pol_map is not None:
                H_cur, W_cur = self.input_resolution
                B = pol_map.shape[0]
                pol_2d = pol_map.transpose(1, 2).view(B, 1, H_cur, W_cur)
                pol_map = F.avg_pool2d(pol_2d, kernel_size=2, stride=2).flatten(2).transpose(1, 2)

        return x, pol_map

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class GlobalContextInjection(nn.Module):
    """Cross-attention from local Stage-0 tokens to global Stage-3 tokens.

    For a 128×128 input with Swin patch_size=8, Stage-0 (16×16) is the only
    stage that uses windowed local attention (4 windows of 8×8).  All deeper
    stages collapse to a single window and are effectively global.  This module
    lets every Stage-0 token attend to the 4 globally-aware Stage-3 tokens,
    injecting scene-level context before the decoder.

    ``gate`` is zero-initialised → tanh(0)=0 → no-op at training start.
    ``proj_out`` is also zero-initialised by ``_reset_pa2_weights``.
    """

    def __init__(self, local_dim: int = 96, global_dim: int = 768, num_heads: int = 4):
        super().__init__()
        assert local_dim % num_heads == 0, "local_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = local_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.norm_local  = nn.LayerNorm(local_dim)
        self.norm_global = nn.LayerNorm(global_dim)
        self.proj_q   = nn.Linear(local_dim,  local_dim, bias=False)
        self.proj_k   = nn.Linear(global_dim, local_dim, bias=False)
        self.proj_v   = nn.Linear(global_dim, local_dim, bias=False)
        self.proj_out = nn.Linear(local_dim,  local_dim, bias=False)
        self.gate     = nn.Parameter(torch.zeros(1))  # tanh(0)=0 → no-op at init

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        """
        local_feat:  (B, L_local,  local_dim)  — Stage-0 tokens, e.g. (B, 256, 96)
        global_feat: (B, L_global, global_dim) — Stage-3 tokens, e.g. (B,   4, 768)
        returns:     (B, L_local,  local_dim)  enriched with global context
        """
        B, L, _ = local_feat.shape
        G        = global_feat.shape[1]
        nH, hd   = self.num_heads, self.head_dim

        Q = self.proj_q(self.norm_local(local_feat))           # (B, L, local_dim)
        K = self.proj_k(self.norm_global(global_feat))          # (B, G, local_dim)
        V = self.proj_v(self.norm_global(global_feat))          # (B, G, local_dim)

        Q = Q.view(B, L, nH, hd).transpose(1, 2)               # (B, nH, L, hd)
        K = K.view(B, G, nH, hd).transpose(1, 2)               # (B, nH, G, hd)
        V = V.view(B, G, nH, hd).transpose(1, 2)               # (B, nH, G, hd)

        attn = (Q @ K.transpose(-2, -1)) * self.scale          # (B, nH, L, G)
        attn = attn.softmax(dim=-1)

        out = (attn @ V).transpose(1, 2).reshape(B, L, -1)     # (B, L, local_dim)
        out = self.proj_out(out)

        return local_feat + self.gate.tanh() * out              # gated residual


class SegformerMLPDecoder(nn.Module):
    """Segformer-style MLP decoder for single-task dense segmentation.

    Each backbone stage's token sequence is independently projected to
    ``embed_dim`` via a Linear+LN+GELU block, reshaped to 2-D spatial maps,
    bilinearly upsampled to the common resolution (input_hw // 4), then
    concatenated along channels, fused by a 1×1 Conv, and finally upsampled
    to the full input resolution.

    Compared with the previous U-Net decoder:
      - No repeated DoubleConv stacks (simpler, fewer parameters).
      - No raw skip-connection noise (Linear+LN acts as a filter).
      - A single bilinear upsample at the end instead of 5-6 stepwise ones.
    """

    def __init__(self,
                 in_channels=(96, 192, 384, 768),
                 embed_dim: int = 256,
                 num_classes: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
            ) for c in in_channels
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        self.head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, features, target_hw):
        """
        features:  list of (B, L_i, C_i) token sequences, one per scale
                   e.g. [ft0:(B,256,96), ft1:(B,64,192), ft2:(B,16,384), ft4:(B,4,768)]
        target_hw: (H, W) of the original input image — output is upsampled here
        returns:   (B, num_classes, H, W) logits
        """
        common_H = target_hw[0] // 4
        common_W = target_hw[1] // 4

        projected = []
        for feat, proj_layer in zip(features, self.proj):
            x = proj_layer(feat)                                    # (B, L, embed_dim)
            B, L, C = x.shape
            H = W = int(L ** 0.5)
            x = x.transpose(1, 2).view(B, C, H, W)                 # (B, C, H, W)
            x = F.interpolate(x, size=(common_H, common_W),
                              mode='bilinear', align_corners=False)
            projected.append(x)

        x = torch.cat(projected, dim=1)                             # (B, C*n, cH, cW)
        x = self.fuse(x)                                            # (B, embed_dim, cH, cW)
        x = F.interpolate(x, size=target_hw,
                          mode='bilinear', align_corners=False)     # (B, embed_dim, H, W)
        return self.head(x)                                         # (B, num_classes, H, W)


class SwinTransformerImproved(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, options, **kwargs):
        super().__init__()
        img_size = options['patch_size']
        patch_size = options['swin_hp']['patch_size']
        default_in_chans = (
            len(options['train_variables'])
            + (1 if options.get('pol_ratio_channel', False) else 0)
            + (2 if options.get('month_encoding', False) else 0)
        )
        in_chans = options.get('n_input_channels', default_in_chans)
        embed_dim = options['swin_hp']['embed_dim']
        depths = options['swin_hp']['depths']
        num_heads = options['swin_hp']['num_heads']
        window_size = options['swin_hp']['window_size']
        mlp_ratio = options['swin_hp']['mlp_ratio']
        qkv_bias = options['swin_hp']['qkv_bias']
        qk_scale = options['swin_hp']['qk_scale']
        drop_rate = options['swin_hp']['drop_rate']
        attn_drop_rate = options['swin_hp']['attn_drop_rate']
        drop_path_rate = options['swin_hp']['drop_path_rate']
        norm_layer = options['swin_hp']['norm_layer']
        ape = options['swin_hp']['ape']
        patch_norm = options['swin_hp']['patch_norm']
        use_checkpoint = options['swin_hp']['use_checkpoint']

        self.pol_ratio_channel = options.get('pol_ratio_channel', False)
        # PA² ablation flags: each controls one of the three improvements independently.
        # Defaults to True (full PA²-Swin). Set False in ablation configs to isolate components.
        self._pa2_kimi   = options.get('pa2_kimi_attnres',  True)
        self._pa2_gate   = options.get('pa2_adaptive_gate', True)
        self._pa2_pol    = options.get('pa2_pol_bias',      True)
        self.net_name = 'SwinTransformerImproved'
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               use_kimi_attnres=(i_layer == 2) and self._pa2_kimi,
                               use_adaptive_gate=self._pa2_gate,
                               use_pol_bias=self._pa2_pol)
            self.layers.append(layer)

 
        # ── Decoder: GlobalContextInjection + Segformer-style MLP Decoder ──────
        # GCI: ft[0] (Stage-0, local 16×16) attends to ft[4] (Stage-3, global 2×2)
        self.gci = GlobalContextInjection(
            local_dim  = embed_dim,             # 96
            global_dim = self.num_features,     # 768
            num_heads  = 4,
        )
        # Segformer MLP decoder: fuses 4-scale features → single SOD output
        self.decoder = SegformerMLPDecoder(
            in_channels = (embed_dim,
                           embed_dim * 2,
                           embed_dim * 4,
                           self.num_features),  # (96, 192, 384, 768)
            embed_dim   = 256,
            num_classes = options['n_classes']['SOD'],
            dropout     = 0.1,
        )

        self.apply(self._init_weights)
        self._reset_pa2_weights()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _reset_pa2_weights(self):
        """Re-zero PA²-specific params after apply(_init_weights) overrides them."""
        for m in self.modules():
            if isinstance(m, WindowAttention):
                nn.init.zeros_(m.alpha_gate[0].weight)
                nn.init.zeros_(m.alpha_gate[0].bias)
                nn.init.zeros_(m.pol_proj.weight)
        # GCI: zero-init output proj so the module is a no-op at training start
        # (gate is already zeros via nn.Parameter(torch.zeros(1)))
        nn.init.zeros_(self.gci.proj_out.weight)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        if self.pol_ratio_channel and self._pa2_pol:
            pol_raw = x[:, -1:, :, :]                                        # (B, 1, H, W)
            pol_map = F.avg_pool2d(pol_raw,
                                   kernel_size=self.patch_embed.patch_size[0],
                                   stride=self.patch_embed.patch_size[0])    # (B, 1, H/p, W/p)
            pol_map = pol_map.flatten(2).transpose(1, 2)                     # (B, L0, 1)
        else:
            pol_map = None

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        ft = [x]
        for layer in self.layers:
            x, pol_map = layer(x, pol_map)
            ft.append(x)

        return ft



    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        ft = self.forward_features(x)
        # ft[0]: (B, L0,  96)  Stage-0, spatial 16x16, local windowed attention
        # ft[1]: (B, L1, 192)  Stage-1, spatial  8x8,  effectively global
        # ft[2]: (B, L2, 384)  Stage-2, spatial  4x4,  global
        # ft[3]: (B, L3, 768)  Stage-2 output after PatchMerging, spatial 2x2
        # ft[4]: (B, L4, 768)  Stage-3, spatial  2x2,  global scene summary

        # Global Context Injection: ft[0] attends to the global ft[4]
        ft0 = self.gci(ft[0], ft[4])

        # Segformer MLP decoder: 4-scale fusion -> SOD logits
        sod = self.decoder([ft0, ft[1], ft[2], ft[4]], target_hw=(H, W))

        return {'SOD': sod}


    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        return flops


#%% Test model

if __name__ == '__main__':

    options = {
        'swin_hp': {
            'patch_size': 8,
            'embed_dim': 96,
            'depths': [2, 2, 6, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 8,
            'mlp_ratio': 4.,
            'qkv_bias': True,
            'qk_scale': None,
            'drop_rate': 0.,
            'attn_drop_rate': 0.,
            'drop_path_rate': 0.1,
            'norm_layer': nn.LayerNorm,
            'ape': False,
            'patch_norm': True,
            'use_checkpoint': False,
        },
        'n_classes': {'SIC': 5, 'SOD': 5, 'FLOE': 5},
        'train_variables': [1, 2, 3, 4],
        'patch_size': 128,
        'pol_ratio_channel': True,
    }

    model = SwinTransformerImproved(options=options)

    # pol_ratio_channel=True → +1 input channel
    x = torch.rand((2, len(options['train_variables']) + 1, options['patch_size'], options['patch_size']))
    out = model(x)

    print('SOD output shape :', out['SOD'].shape)   # expect (2, 5, 128, 128)
    total = sum(p.numel() for p in model.parameters())
    print(f'Total parameters  : {total / 1e6:.2f}M')
