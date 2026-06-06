#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
脚本功能：
1. 载入指定的配置文件和权重（由占位符指定）。
2. 在验证集（或测试集）上逐场景进行推理。
3. 计算每个场景的指标（如 F1, mIoU, OA），并保存为 CSV。
4. 汇总所有场景的像素级预测，计算整个数据集的最终全局指标，并打印和保存总体结果。
"""

import argparse
import csv
import json
import os
import os.path as osp
from collections import OrderedDict
import torch
from mmcv import Config
from torchmetrics.functional import f1_score, jaccard_index
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassF1Score, MulticlassJaccardIndex, MulticlassAccuracy
from tqdm import tqdm

# 根据具体项目调整导入路径
from src.functions import class_decider, get_model, slide_inference
from src.loaders import AI4ArcticChallengeTestDataset, get_variable_options


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate per-scene metrics and overall metrics on validation/test set."
    )
    # 使用占位符作为默认值，用户在 AutoDL 运行时可以通过命令行参数覆盖
    parser.add_argument("--config", type=str, default="<YOUR_CONFIG_PATH>", help="Path to config file.")
    parser.add_argument("--checkpoint", type=str, default="<YOUR_CHECKPOINT_PATH>", help="Path to .pth checkpoint.")
    parser.add_argument("--val-list", type=str, default="<YOUR_VAL_LIST_JSON>", help="Path to validation/test list JSON.")
    parser.add_argument("--data-root", type=str, default="<YOUR_DATA_ROOT>", help="Data root containing .nc files.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for evaluation.")
    parser.add_argument("--output-csv", type=str, default="./per_scene_metrics.csv", help="Output path for per-scene metrics.")
    parser.add_argument("--output-overall", type=str, default="./overall_metrics.txt", help="Output path for overall metrics.")
    return parser.parse_args()


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in ckpt.keys() and isinstance(ckpt[key], dict):
                return ckpt[key]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Unsupported checkpoint format.")


def run_scene_inference(net, train_options, inf_x, mode="val"):
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
        model_sel = train_options.get("model_selection", "").lower()
        # 只要是需要切块滑窗推理的模型（Swin系列、PA2系列等），都使用 slide_inference
        if "swin" in model_sel or "pa2" in model_sel:
            output = slide_inference(inf_x, net, train_options, mode)
        else:
            output = net(inf_x)
    return output


def main():
    args = parse_args()
    
    # 1. 载入配置和数据列表
    cfg = Config.fromfile(args.config)
    train_options = get_variable_options(cfg.train_options)
    
    with open(args.val_list, "r", encoding="utf-8") as f:
        val_scenes = json.load(f)
        
    train_options["path_to_train_data"] = args.data_root
    train_options["path_to_test_data"] = args.data_root

    device = torch.device(args.device)

    # 2. 构建模型并加载权重
    net = get_model(train_options, device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = extract_state_dict(ckpt)
    
    # 尝试去掉可能的多卡 prefix
    normalized_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_k = k.replace("module.", "").replace("_orig_mod.", "")
        normalized_state_dict[new_k] = v
        
    net.load_state_dict(normalized_state_dict, strict=True)
    net.eval()

    # 3. 准备 Dataloader
    dataset = AI4ArcticChallengeTestDataset(
        options=train_options,
        files=val_scenes,
        mode="train",  # 使用 train 模式以返回真实标签和 valid mask
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=train_options.get("num_workers_val", 4),
        shuffle=False,
    )

    chart = train_options.get("target_chart", "SOD")
    n_classes = train_options["n_classes"][chart]

    # 初始化全局 Metric 追踪器
    global_f1 = MulticlassF1Score(num_classes=n_classes, average="weighted").to(device)
    global_miou = MulticlassJaccardIndex(num_classes=n_classes).to(device)
    global_acc = MulticlassAccuracy(num_classes=n_classes).to(device)

    per_scene_rows = []
    
    print(f"Start evaluating on {len(val_scenes)} scenes...")

    # 4. 逐个场景进行推理
    for inf_x, inf_y, cfv_masks, _, scene_name, original_size in tqdm(dataloader):
        scene_name = osp.splitext(scene_name)[0]
        inf_x = inf_x.to(device, non_blocking=True)
        
        output = run_scene_inference(net, train_options, inf_x, mode="val")
        
        # 还原输入尺寸与 Mask
        masks_int = cfv_masks[chart].to(torch.uint8)
        masks_int = torch.nn.functional.interpolate(
            masks_int.unsqueeze(0).unsqueeze(0),
            size=original_size,
            mode="nearest",
        ).squeeze().squeeze()
        cfv_masks[chart] = torch.gt(masks_int, 0)

        # 还原输出尺寸
        if train_options.get("down_sample_scale", 1) != 1:
            if output[chart].size(3) == 1:
                output[chart] = output[chart].permute(0, 3, 1, 2)
                output[chart] = torch.nn.functional.interpolate(
                    output[chart], size=original_size, mode="nearest"
                )
                output[chart] = output[chart].permute(0, 2, 3, 1)
            else:
                output[chart] = torch.nn.functional.interpolate(
                    output[chart], size=original_size, mode="nearest"
                )

        inf_y[chart] = torch.nn.functional.interpolate(
            inf_y[chart].unsqueeze(dim=0).unsqueeze(dim=0),
            size=original_size,
            mode="nearest",
        ).squeeze()

        # 对齐 ground-truth 与 prediction，过滤 invalid pixels (e.g. land)
        pred = class_decider(output[chart], train_options, chart).detach().long()
        true = inf_y[chart].long().to(device, non_blocking=True)
        valid = (~cfv_masks[chart]).to(device)

        pred_flat = pred[valid]
        true_flat = true[valid]
        
        valid_pixels = int(true_flat.numel())
        row = {"scene": scene_name, "valid_pixels": valid_pixels}

        if valid_pixels > 0:
            # 记录用于全局计算
            global_f1.update(pred_flat, true_flat)
            global_miou.update(pred_flat, true_flat)
            global_acc.update(pred_flat, true_flat)
            
            # 场景内指标
            scene_f1 = f1_score(target=true_flat, preds=pred_flat, average="weighted", task="multiclass", num_classes=n_classes)
            scene_miou = jaccard_index(target=true_flat, preds=pred_flat, task="multiclass", num_classes=n_classes)
            scene_oa = (true_flat == pred_flat).float().mean()
            
            row["f1_percent"] = round(scene_f1.item() * 100, 3)
            row["mIoU_percent"] = round(scene_miou.item() * 100, 3)
            row["OA_percent"] = round(scene_oa.item() * 100, 3)
        else:
            row["f1_percent"] = float("nan")
            row["mIoU_percent"] = float("nan")
            row["OA_percent"] = float("nan")

        per_scene_rows.append(row)

    # 5. 保存每个场景结果
    os.makedirs(osp.dirname(osp.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scene", "valid_pixels", "f1_percent", "mIoU_percent", "OA_percent"])
        writer.writeheader()
        writer.writerows(per_scene_rows)

    # 6. 计算和输出总体指标
    final_f1 = global_f1.compute().item() * 100
    final_miou = global_miou.compute().item() * 100
    final_acc = global_acc.compute().item() * 100

    overall_res = (
        "================ Final Overall Metrics ================\n"
        f"Target Chart: {chart}\n"
        f"Global F1 Score: {final_f1:.3f}%\n"
        f"Global mIoU:     {final_miou:.3f}%\n"
        f"Global Overall Accuracy: {final_acc:.3f}%\n"
        "=======================================================\n"
    )
    print(overall_res)
    
    os.makedirs(osp.dirname(osp.abspath(args.output_overall)), exist_ok=True)
    with open(args.output_overall, "w", encoding="utf-8") as f:
        f.write(overall_res)
        
    print(f"Success! Per-scene metrics saved to: {args.output_csv}")
    print(f"Overall metrics saved to: {args.output_overall}")

if __name__ == "__main__":
    main()