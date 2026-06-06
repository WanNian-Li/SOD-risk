"""
MMSegWrapper: 将 mmsegmentation 的 backbone+head 适配到 MySeaIce 项目接口。

接口约定
--------
- 输入 : Tensor [B, C, H, W]，C 由 train_options['train_variables'] 决定
- 输出 : dict {chart_name: Tensor [B, n_classes, H, W]}
          例如 {'SOD': Tensor [B, 5, 256, 256]}

Config 示例（在 train_options 里添加）
--------------------------------------
    'model_selection': 'mmseg',

    'mmseg_backbone': {
        'type': 'MixVisionTransformer',  # mmseg backbone 类名
        # 其余所有参数直接传给该类的 __init__
        # in_channels 会被自动注入，无需手填
    },

    'mmseg_decode_head': {
        'type': 'SegformerHead',  # mmseg decode head 类名
        # num_classes 会被自动注入，无需手填
        'in_channels': [32, 64, 160, 256],
        'in_index': [0, 1, 2, 3],
        'channels': 256,
        'dropout_ratio': 0.1,
        'align_corners': False,
    },

支持的 backbone（直接从 mmseg.models.backbones 导入）:
    MixVisionTransformer, ResNet, SwinTransformer, MSCAN, ConvNeXt, ...

支持的 decode head（直接从 mmseg.models.decode_heads 导入）:
    SegformerHead, UPerHead, FCNHead, ASPPHead,
    DepthwiseSeparableASPPHead, LightHamHead, ...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 部分 backbone 的输入通道参数名不叫 in_channels，需要特殊处理
# --------------------------------------------------------------------------- #
_BACKBONE_IN_CH_KWARG = {
    # backbone type  ->  参数名
    'MixVisionTransformer': 'in_channels',
    'ResNet':               'in_channels',   # 实际上 ResNet 固定 3ch，需魔改；见下文注释
    'SwinTransformer':      'in_channels',
    'MSCAN':                'in_channels',
    'ConvNeXt':             'in_channels',
    # 如有新模型在此追加
}

# 不支持直接设置 in_channels 的 backbone（通常只支持 RGB 3ch），
# 对这类 backbone 用 1×1 Conv 做输入适配层。
_BACKBONE_NO_IN_CH = {'ResNet'}


class MMSegWrapper(nn.Module):
    """
    将任意 mmseg backbone + decode head 组合包装为 MySeaIce 项目接口。

    Parameters
    ----------
    options : dict
        train_options，必须包含：
          - train_variables       : list，用于推断输入通道数
          - charts                : list，输出的任务名列表，例如 ['SOD']
          - n_classes             : dict，如 {'SOD': 5}
          - mmseg_backbone        : dict，含 'type' 及 backbone 参数
          - mmseg_decode_head     : dict，含 'type' 及 head 参数
        可选：
          - month_encoding        : bool，若 True 则输入通道 +2
          - pol_ratio_channel     : bool，若 True 则输入通道 +1
    """

    def __init__(self, options: dict):
        super().__init__()

        # ------------------------------------------------------------------ #
        # 1. 推断实际输入通道数
        # ------------------------------------------------------------------ #
        in_channels = len(options['train_variables'])
        if options.get('month_encoding', False):
            in_channels += 2
        if options.get('pol_ratio_channel', False):
            in_channels += 1

        self.charts = options['charts']
        self.n_classes = options['n_classes']

        # ------------------------------------------------------------------ #
        # 2. 构建 backbone
        # ------------------------------------------------------------------ #
        backbone_cfg = options['mmseg_backbone'].copy()
        backbone_type = backbone_cfg.pop('type')

        self.backbone, self.input_adapter = self._build_backbone(
            backbone_type, backbone_cfg, in_channels
        )
        self.backbone.init_weights()  # mmseg backbone 统一有 init_weights()

        # ------------------------------------------------------------------ #
        # 3. 为每个 chart 构建独立的 decode head（共享 backbone）
        # ------------------------------------------------------------------ #
        head_cfg_template = options['mmseg_decode_head'].copy()
        head_type = head_cfg_template.pop('type')

        self.decode_heads = nn.ModuleDict()
        for chart in self.charts:
            cfg = head_cfg_template.copy()
            cfg['num_classes'] = self.n_classes[chart]
            self.decode_heads[chart] = self._build_head(head_type, cfg)

    # ---------------------------------------------------------------------- #
    # 内部构建方法
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _build_backbone(backbone_type: str, cfg: dict, in_channels: int):
        """
        返回 (backbone, input_adapter)。
        - 若 backbone 支持 in_channels 参数，直接设置；input_adapter = nn.Identity()
        - 若不支持（如标准 ResNet），用 1×1 Conv 把输入映射到 3 通道；input_adapter = Conv
        """
        try:
            from mmseg.models import backbones as mmseg_backbones
        except ImportError as e:
            raise ImportError(
                "找不到 mmsegmentation。请在 AutoDL 上执行：\n"
                "  pip install -U openmim && mim install mmengine 'mmcv>=2.0.0'\n"
                "  pip install mmsegmentation"
            ) from e

        if backbone_type not in _BACKBONE_NO_IN_CH:
            # 直接注入 in_channels
            in_ch_kwarg = _BACKBONE_IN_CH_KWARG.get(backbone_type, 'in_channels')
            cfg[in_ch_kwarg] = in_channels
            backbone = getattr(mmseg_backbones, backbone_type)(**cfg)
            adapter = nn.Identity()
        else:
            # 不支持非 3 通道的 backbone：在外部加一个 1×1 Conv 适配
            backbone = getattr(mmseg_backbones, backbone_type)(**cfg)
            adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)

        return backbone, adapter

    @staticmethod
    def _build_head(head_type: str, cfg: dict):
        try:
            from mmseg.models import decode_heads as mmseg_heads
        except ImportError as e:
            raise ImportError("找不到 mmsegmentation。") from e

        return getattr(mmseg_heads, head_type)(**cfg)

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        x : Tensor [B, C, H, W]

        Returns
        -------
        dict {chart: Tensor [B, n_classes, H, W]}
        """
        # 输入适配（通常是 Identity，仅 ResNet 等需要 3ch 时有效）
        x_bb = self.input_adapter(x)

        # Backbone 前向 → 多尺度特征 tuple
        features = self.backbone(x_bb)  # e.g. (f0, f1, f2, f3)

        output = {}
        for chart in self.charts:
            logits = self.decode_heads[chart](features)  # [B, n_cls, H', W']
            # 上采样回输入分辨率
            if logits.shape[-2:] != x.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=x.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
            output[chart] = logits

        return output
