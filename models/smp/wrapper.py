"""
SMPWrapper: 将 segmentation_models_pytorch (SMP) 模型适配到 MySeaIce 项目接口。

接口约定
--------
- 输入 : Tensor [B, C, H, W]，C 由 train_options['train_variables'] 自动推断
- 输出 : dict {chart_name: Tensor [B, n_classes, H, W]}
          例如 {'SOD': Tensor [B, 5, 256, 256]}

Config 示例（在 train_options 里添加）
--------------------------------------
    'model_selection': 'smp',

    'smp_model': {
        'arch': 'DeepLabV3Plus',    # SMP 架构类名（见下方支持列表）
        'encoder_name': 'resnet50', # 编码器名称
        'encoder_weights': None,    # None=随机初始化；'imagenet'=ImageNet预训练
        # 其余参数原样传给对应 SMP 类的 __init__（见各架构文档）
    },

支持的架构（arch 字段可选值）:
    Unet, UnetPlusPlus, MAnet, Linknet,
    FPN, PSPNet, PAN, DeepLabV3, DeepLabV3Plus

常用编码器（encoder_name 字段可选值）:
    resnet50, resnet101
    efficientnet-b4, efficientnet-b7
    mit_b0 ~ mit_b5          (Mix Transformer, Segformer 同款)
    swin_tiny_patch4_window7_224  (需要安装 timm)
    convnext_tiny, convnext_small (需要安装 timm)

注意
----
- encoder_weights='imagenet' 时 SMP 会自动把预训练权重适配到 in_channels≠3 的情况
  （将3通道权重按均值扩展到N通道），对遥感多通道输入有一定迁移效果。
- 当 charts 有多个任务时，每个任务独享一个 SMP 解码头，共享编码器权重（节省显存）。
  目前仅 SOD 一个任务，等同于单任务模型。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SMPWrapper(nn.Module):
    """
    将任意 SMP 架构包装为 MySeaIce 项目接口。

    Parameters
    ----------
    options : dict
        train_options，必须包含：
          - train_variables : list，用于推断输入通道数
          - charts          : list，任务名列表，如 ['SOD']
          - n_classes       : dict，如 {'SOD': 5}
          - smp_model       : dict，含 'arch'、'encoder_name' 及其他 SMP 参数
        可选：
          - month_encoding    : bool，若 True 输入通道 +2
          - pol_ratio_channel : bool，若 True 输入通道 +1
    """

    def __init__(self, options: dict):
        super().__init__()

        try:
            import segmentation_models_pytorch as smp
        except ImportError as e:
            raise ImportError(
                "找不到 segmentation_models_pytorch。请执行：\n"
                "  pip install segmentation-models-pytorch"
            ) from e

        # ------------------------------------------------------------------ #
        # 1. 推断输入通道数
        # ------------------------------------------------------------------ #
        in_channels = len(options['train_variables'])
        if options.get('month_encoding', False):
            in_channels += 2
        if options.get('pol_ratio_channel', False):
            in_channels += 1

        self.charts = options['charts']
        self.n_classes = options['n_classes']

        # ------------------------------------------------------------------ #
        # 2. 解析 SMP 配置
        # ------------------------------------------------------------------ #
        smp_cfg = options['smp_model'].copy()
        arch = smp_cfg.pop('arch')           # 从 kwargs 里摘出架构名
        model_cls = getattr(smp, arch, None)
        if model_cls is None:
            raise ValueError(
                f"SMP 中找不到架构 '{arch}'。"
                f"可用架构：Unet, UnetPlusPlus, MAnet, Linknet, "
                f"FPN, PSPNet, PAN, DeepLabV3, DeepLabV3Plus"
            )

        # ------------------------------------------------------------------ #
        # 3. 为每个 chart 构建独立模型
        #    当前只有 SOD 一个任务，等同单任务模型
        # ------------------------------------------------------------------ #
        self.models = nn.ModuleDict()
        for chart in self.charts:
            self.models[chart] = model_cls(
                in_channels=in_channels,
                classes=self.n_classes[chart],
                **smp_cfg,
            )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        x : Tensor [B, C, H, W]

        Returns
        -------
        dict {chart: Tensor [B, n_classes, H, W]}
        """
        output = {}
        for chart in self.charts:
            logits = self.models[chart](x)   # SMP 输出 [B, n_cls, H, W]
            # 部分架构（如 PSPNet）输出尺寸可能与输入不同，统一上采样
            if logits.shape[-2:] != x.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=x.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
            output[chart] = logits
        return output
