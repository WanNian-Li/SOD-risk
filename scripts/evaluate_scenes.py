#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-scene evaluation script. Supports single-model and multi-model ensemble inference.

When multiple --config / --checkpoint pairs are provided the logits from all models
are averaged before class decoding (logit-level ensemble).

Usage:
    # Single model
    python evaluate_scenes.py \
        --config configs/SOD/smp_fpn_mit_b2.py \
        --checkpoint work_dirs/fold0/best_model.pth \
        --val-list datalists/cv_folds6/fold0_val.json

    # Multi-model ensemble
    python evaluate_scenes.py \
        --config configs/SOD/model_a.py configs/SOD/model_b.py \
        --checkpoint work_dirs/fold0/best_a.pth work_dirs/fold1/best_b.pth \
        --val-list datalists/cv_folds6/fold0_val.json \
        --output-dir eval_results/ensemble

All models must share the same charts and n_classes.
Dataset settings (val_path, variable options, etc.) are taken from the first config.
"""

import argparse
import copy
import json
import os
import os.path as osp
import pathlib
import sys
import warnings
from collections import defaultdict

import numpy as np
import torch
from mmcv import Config, mkdir_or_exist
from tqdm import tqdm

warnings.filterwarnings("ignore")

from src.functions import (
    batched_slide_inference,
    class_decider,
    compute_class_proportions,
    compute_classwise_f1score,
    compute_classwise_IoU,
    compute_classwise_precision_recall,
    compute_mIoU,
    compute_metrics,
    compute_normalized_confusion_matrix,
    compute_ordinal_cost_metric,
    compute_overall_accuracy,
    fast_tiled_val_inference,
    get_model,
    slide_inference,
)
from src.loaders import AI4ArcticChallengeTestDataset, get_variable_options
from src.utils import GROUP_NAMES


def parse_args():
    parser = argparse.ArgumentParser(
        description='Per-scene evaluation. Accepts one or more config/checkpoint pairs for ensemble inference.')
    parser.add_argument('--config', type=pathlib.Path, required=True, nargs='+',
                        help='One or more training config file paths. '
                             'Must match the number of --checkpoint arguments.')
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True, nargs='+',
                        help='One or more trained .pth checkpoint paths. '
                             'Must match the number of --config arguments.')
    parser.add_argument('--val-list', type=pathlib.Path, required=True,
                        help='Path to the validation JSON list (e.g. datalists/cv_folds6/fold0_val.json).')
    parser.add_argument('--output-dir', type=pathlib.Path, default=None,
                        help='Directory to save per-scene result files. '
                             'Defaults to <first_checkpoint_dir>/eval_scenes.')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU device ID (default: 0).')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to disk.')
    return parser.parse_args()


def load_checkpoint(net, checkpoint_path, device):
    """Load model weights, handling compile-mode '_orig_mod.' prefix."""
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}
    net.load_state_dict(state_dict)
    print(f'Checkpoint loaded from {checkpoint_path}')
    return net


def build_val_loader(train_options, val_list_path):
    """Build a DataLoader for the validation set."""
    val_files = json.loads(open(str(val_list_path)).read())
    dataset = AI4ArcticChallengeTestDataset(
        options=train_options, files=val_files, mode='train')
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=None,
        num_workers=train_options.get('num_workers_val', 0),
        shuffle=False)
    return loader, val_files


def _ensure_tensor(x):
    """Convert numpy array or tensor to a CPU tensor (copy)."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.copy())
    elif isinstance(x, torch.Tensor):
        return x.clone().cpu()
    else:
        return torch.as_tensor(x)


def compute_scene_metrics(output_class, inf_y, cfv_masks, train_options):
    """Compute all metrics for a single scene (flattened valid pixels)."""
    charts = train_options['charts']
    n_classes = train_options['n_classes']
    device = output_class[charts[0]].device

    outs_flat = {}
    trues_flat = {}
    for chart in charts:
        pred = output_class[chart].clone().to(device)
        true = _ensure_tensor(inf_y[chart]).long().to(device)
        outs_flat[chart] = pred[~cfv_masks[chart]]
        trues_flat[chart] = true[~cfv_masks[chart]]

    # Guard: if any chart has no valid pixels, return NaN results and skip torchmetrics
    if any(trues_flat[c].numel() == 0 for c in charts):
        n_cls_list = {c: n_classes[c] for c in charts}
        nan_per_class = {c: [float('nan')] * n_classes[c] for c in charts}
        return {
            'miou':                 {c: float('nan') for c in charts},
            'iou_per_class':        nan_per_class,
            'f1_weighted':          {c: float('nan') for c in charts},
            'combined_score':       float('nan'),
            'f1_per_class':         {c: [float('nan')] * n_classes[c] for c in charts},
            'oa':                   {c: float('nan') for c in charts},
            'precision_per_class':  {c: [float('nan')] * n_classes[c] for c in charts},
            'recall_per_class':     {c: [float('nan')] * n_classes[c] for c in charts},
            'confmat_norm':         {c: [[0.0] * n_classes[c]] * n_classes[c] for c in charts},
            'true_proportions':     {c: [float('nan')] * n_classes[c] for c in charts},
            'pred_proportions':     {c: [float('nan')] * n_classes[c] for c in charts},
        }

    results = {}

    miou = compute_mIoU(trues_flat, outs_flat, charts, n_classes)
    results['miou'] = {c: miou[c].item() for c in charts}

    classwise_iou = compute_classwise_IoU(trues_flat, outs_flat, charts, n_classes)
    results['iou_per_class'] = {c: [v.item() for v in classwise_iou[c]] for c in charts}

    combined_score, scores = compute_metrics(
        trues_flat, outs_flat, charts,
        train_options['chart_metric'], n_classes)
    results['f1_weighted'] = {c: scores[c].item() for c in charts}
    results['combined_score'] = combined_score.item()

    classwise_f1 = compute_classwise_f1score(trues_flat, outs_flat, charts, n_classes)
    results['f1_per_class'] = {c: [v.item() for v in classwise_f1[c]] for c in charts}

    oa = compute_overall_accuracy(trues_flat, outs_flat, charts)
    results['oa'] = {c: oa[c] for c in charts}

    prec, rec = compute_classwise_precision_recall(trues_flat, outs_flat, charts, n_classes)
    results['precision_per_class'] = {c: [v.item() for v in prec[c]] for c in charts}
    results['recall_per_class'] = {c: [v.item() for v in rec[c]] for c in charts}

    cm_norm = compute_normalized_confusion_matrix(trues_flat, outs_flat, charts, n_classes)
    results['confmat_norm'] = {c: cm_norm[c].cpu().numpy().tolist() for c in charts}

    true_props, pred_props = compute_class_proportions(trues_flat, outs_flat, charts, n_classes)
    results['true_proportions'] = {c: [v.item() for v in true_props[c]] for c in charts}
    results['pred_proportions'] = {c: [v.item() for v in pred_props[c]] for c in charts}

    ordinal_cfg = train_options.get('ordinal_metric', {})
    if ordinal_cfg.get('enabled', False):
        raw_costs, norm_scores = compute_ordinal_cost_metric(
            trues_flat, outs_flat, charts, n_classes,
            cost_per_distance=ordinal_cfg.get('cost_per_distance', None))
        results['ordinal_mean_cost'] = {c: raw_costs[c].item() for c in charts}
        results['ordinal_score'] = {c: norm_scores[c].item() for c in charts}

    return results


def _update_confusion_matrix(cm_accum, pred, true, num_classes):
    """Incrementally update a confusion matrix from flat pred/true tensors (both 1D, long).

    Returns updated cm_accum of shape (num_classes, num_classes).
    """
    valid = ((true >= 0) & (true < num_classes) &
             (pred >= 0) & (pred < num_classes))
    t = true[valid].view(-1)
    p = pred[valid].view(-1)
    idx = t * num_classes + p
    binc = torch.bincount(idx, minlength=num_classes * num_classes)
    cm_accum += binc.view(num_classes, num_classes).float()
    return cm_accum


def derive_metrics_from_cm(global_cm, train_options):
    """Derive all global metrics from accumulated confusion matrices (no pixel storage)."""
    charts = train_options['charts']
    n_classes = train_options['n_classes']
    target_chart = train_options.get('target_chart', charts[0])

    results = {}

    for chart in charts:
        cm = global_cm[chart]  # (n_classes, n_classes), float
        n_c = n_classes[chart]

        tp = torch.diag(cm)
        pred_sum = cm.sum(dim=0)
        true_sum = cm.sum(dim=1)
        total = cm.sum().clamp(min=1.0)

        # Per-class IoU
        iou_per_class = tp / (pred_sum + true_sum - tp).clamp(min=1e-8)
        # mIoU
        results.setdefault('miou', {})[chart] = iou_per_class.mean().item()
        results.setdefault('iou_per_class', {})[chart] = iou_per_class.tolist()

        # Per-class Precision / Recall
        precision = tp / pred_sum.clamp(min=1e-8)
        recall = tp / true_sum.clamp(min=1e-8)
        results.setdefault('precision_per_class', {})[chart] = precision.tolist()
        results.setdefault('recall_per_class', {})[chart] = recall.tolist()

        # Per-class F1
        f1_per_class = 2 * tp / (pred_sum + true_sum).clamp(min=1e-8)
        results.setdefault('f1_per_class', {})[chart] = f1_per_class.tolist()

        # Weighted F1 (weights = true class proportion), scaled to [0, 100]
        # to match compute_metrics() output convention.
        weights = true_sum / total
        f1_weighted_raw = (f1_per_class * weights).sum()
        results.setdefault('f1_weighted', {})[chart] = round(f1_weighted_raw.item() * 100.0, 3)

        # OA
        oa = (tp.sum() / total).item()
        results.setdefault('oa', {})[chart] = oa

        # Normalized confusion matrix (row-normalized)
        cm_norm = cm / true_sum.clamp(min=1e-8).unsqueeze(1)
        results.setdefault('confmat_norm', {})[chart] = cm_norm.tolist()

        # Class proportions
        true_props = true_sum / total
        pred_props = pred_sum / total
        results.setdefault('true_proportions', {})[chart] = true_props.tolist()
        results.setdefault('pred_proportions', {})[chart] = pred_props.tolist()

        # Ordinal cost from confusion matrix
        ordinal_cfg = train_options.get('ordinal_metric', {})
        if ordinal_cfg.get('enabled', False):
            cost_table = (ordinal_cfg.get('cost_per_distance', None) or
                          list(range(n_c)))
            cost_table = torch.tensor(cost_table, dtype=torch.float32, device=cm.device)
            # mean_cost = sum_{i,j} cm[i,j] * cost_table[|i-j|] / total
            cost_sum = 0.0
            for i in range(n_c):
                for j in range(n_c):
                    cost_sum += cm[i, j].item() * cost_table[abs(i - j)].item()
            mean_cost = cost_sum / total.item()
            max_cost = float(cost_table.max())
            results.setdefault('ordinal_mean_cost', {})[chart] = mean_cost
            results.setdefault('ordinal_score', {})[chart] = 1.0 - mean_cost / max(max_cost, 1e-6)

    # Combined score (weighted average of per-chart metrics, already in [0,100] scale)
    metrics_cfg = train_options['chart_metric']
    combined = 0.0
    sum_w = 0.0
    for chart in charts:
        w = metrics_cfg[chart]['weight']
        combined += results['f1_weighted'][chart] * w
        sum_w += w
    results['combined_score'] = round(combined / max(sum_w, 1e-8), 3)

    return results


def aggregate_global_metrics(outputs_flat, inf_ys_flat, train_options):
    """Compute aggregated metrics across all scenes (legacy, use derive_metrics_from_cm instead)."""
    charts = train_options['charts']
    n_classes = train_options['n_classes']

    results = {}

    miou = compute_mIoU(inf_ys_flat, outputs_flat, charts, n_classes)
    results['miou'] = {c: miou[c].item() for c in charts}

    classwise_iou = compute_classwise_IoU(inf_ys_flat, outputs_flat, charts, n_classes)
    results['iou_per_class'] = {c: [v.item() for v in classwise_iou[c]] for c in charts}

    combined_score, scores = compute_metrics(
        inf_ys_flat, outputs_flat, charts,
        train_options['chart_metric'], n_classes)
    results['f1_weighted'] = {c: scores[c].item() for c in charts}
    results['combined_score'] = combined_score.item()

    classwise_f1 = compute_classwise_f1score(inf_ys_flat, outputs_flat, charts, n_classes)
    results['f1_per_class'] = {c: [v.item() for v in classwise_f1[c]] for c in charts}

    oa = compute_overall_accuracy(inf_ys_flat, outputs_flat, charts)
    results['oa'] = {c: oa[c] for c in charts}

    prec, rec = compute_classwise_precision_recall(inf_ys_flat, outputs_flat, charts, n_classes)
    results['precision_per_class'] = {c: [v.item() for v in prec[c]] for c in charts}
    results['recall_per_class'] = {c: [v.item() for v in rec[c]] for c in charts}

    cm_norm = compute_normalized_confusion_matrix(inf_ys_flat, outputs_flat, charts, n_classes)
    results['confmat_norm'] = {c: cm_norm[c].cpu().numpy().tolist() for c in charts}

    true_props, pred_props = compute_class_proportions(inf_ys_flat, outputs_flat, charts, n_classes)
    results['true_proportions'] = {c: [v.item() for v in true_props[c]] for c in charts}
    results['pred_proportions'] = {c: [v.item() for v in pred_props[c]] for c in charts}

    ordinal_cfg = train_options.get('ordinal_metric', {})
    if ordinal_cfg.get('enabled', False):
        raw_costs, norm_scores = compute_ordinal_cost_metric(
            inf_ys_flat, outputs_flat, charts, n_classes,
            cost_per_distance=ordinal_cfg.get('cost_per_distance', None))
        results['ordinal_mean_cost'] = {c: raw_costs[c].item() for c in charts}
        results['ordinal_score'] = {c: norm_scores[c].item() for c in charts}

    return results


def print_scene_result(scene_name, result, train_options):
    """Pretty-print metrics for a single scene."""
    target_chart = train_options.get('target_chart', train_options['charts'][0])
    n_classes = train_options['n_classes'][target_chart]
    class_names = list(GROUP_NAMES.get(target_chart, {}).values()) if target_chart in GROUP_NAMES else [str(i) for i in range(n_classes)]

    print(f"\n{'='*70}")
    print(f"  Scene: {scene_name}")
    print(f"{'='*70}")

    print(f"  {target_chart} mIoU:        {result['miou'][target_chart]:.4f}")
    print(f"  {target_chart} OA:          {result['oa'][target_chart]:.4f}")
    print(f"  {target_chart} F1 (weighted): {result['f1_weighted'][target_chart]:.4f}")
    print(f"  Combined score:              {result['combined_score']:.4f}")

    print(f"\n  Per-class metrics ({target_chart}):")
    header = f"  {'Class':<20s} {'IoU':>8s} {'F1':>8s} {'Prec':>8s} {'Rec':>8s} {'True%':>8s} {'Pred%':>8s}"
    print(header)
    print("  " + "-" * len(header))
    for c in range(n_classes):
        cname = class_names[c] if c < len(class_names) else f"cls{c}"
        print(f"  {cname:<20s} "
              f"{result['iou_per_class'][target_chart][c]:8.4f} "
              f"{result['f1_per_class'][target_chart][c]:8.4f} "
              f"{result['precision_per_class'][target_chart][c]:8.4f} "
              f"{result['recall_per_class'][target_chart][c]:8.4f} "
              f"{result['true_proportions'][target_chart][c]:8.4f} "
              f"{result['pred_proportions'][target_chart][c]:8.4f}")

    if 'ordinal_score' in result:
        print(f"\n  Ordinal Mean Cost:  {result['ordinal_mean_cost'][target_chart]:.4f}")
        print(f"  Ordinal Score:      {result['ordinal_score'][target_chart]:.4f}")

    print(f"\n  Normalized Confusion Matrix ({target_chart}, row=true):")
    cm = result['confmat_norm'][target_chart]
    row_labels = class_names[:n_classes]
    col_width = max(8, max(len(l) for l in row_labels))
    hdr = " " * (max(len(l) for l in row_labels) + 4) + "".join(f"{l:>{col_width}s}" for l in row_labels)
    print(hdr)
    for r, rname in enumerate(row_labels):
        vals = "".join(f"{cm[r][c]:{col_width}.4f}" for c in range(n_classes))
        print(f"  {rname:<{max(len(l) for l in row_labels)}s}  {vals}")


def print_global_result(result, train_options):
    """Pretty-print aggregated metrics."""
    target_chart = train_options.get('target_chart', train_options['charts'][0])
    n_classes = train_options['n_classes'][target_chart]
    class_names = list(GROUP_NAMES.get(target_chart, {}).values()) if target_chart in GROUP_NAMES else [str(i) for i in range(n_classes)]

    print(f"\n{'#'*70}")
    print(f"  GLOBAL RESULTS (aggregated over all scenes)")
    print(f"{'#'*70}")

    print(f"  {target_chart} mIoU:        {result['miou'][target_chart]:.4f}")
    print(f"  {target_chart} OA:          {result['oa'][target_chart]:.4f}")
    print(f"  {target_chart} F1 (weighted): {result['f1_weighted'][target_chart]:.4f}")
    print(f"  Combined score:              {result['combined_score']:.4f}")

    print(f"\n  Per-class metrics ({target_chart}):")
    header = f"  {'Class':<20s} {'IoU':>8s} {'F1':>8s} {'Prec':>8s} {'Rec':>8s} {'True%':>8s} {'Pred%':>8s}"
    print(header)
    print("  " + "-" * len(header))
    for c in range(n_classes):
        cname = class_names[c] if c < len(class_names) else f"cls{c}"
        print(f"  {cname:<20s} "
              f"{result['iou_per_class'][target_chart][c]:8.4f} "
              f"{result['f1_per_class'][target_chart][c]:8.4f} "
              f"{result['precision_per_class'][target_chart][c]:8.4f} "
              f"{result['recall_per_class'][target_chart][c]:8.4f} "
              f"{result['true_proportions'][target_chart][c]:8.4f} "
              f"{result['pred_proportions'][target_chart][c]:8.4f}")

    if 'ordinal_score' in result:
        print(f"\n  Ordinal Mean Cost:  {result['ordinal_mean_cost'][target_chart]:.4f}")
        print(f"  Ordinal Score:      {result['ordinal_score'][target_chart]:.4f}")

    print(f"\n  Normalized Confusion Matrix ({target_chart}, row=true):")
    cm = result['confmat_norm'][target_chart]
    row_labels = class_names[:n_classes]
    col_width = max(8, max(len(l) for l in row_labels))
    hdr = " " * (max(len(l) for l in row_labels) + 4) + "".join(f"{l:>{col_width}s}" for l in row_labels)
    print(hdr)
    for r, rname in enumerate(row_labels):
        vals = "".join(f"{cm[r][c]:{col_width}.4f}" for c in range(n_classes))
        print(f"  {rname:<{max(len(l) for l in row_labels)}s}  {vals}")


def _infer_single(inf_x, net, train_options):
    """Run inference with one model, choosing the correct strategy from its train_options."""
    if train_options['model_selection'] in ('swin', 'PA2'):
        return batched_slide_inference(inf_x, net, train_options, 'val')
    elif (inf_x.shape[2] > train_options['patch_size'] or
          inf_x.shape[3] > train_options['patch_size']):
        return fast_tiled_val_inference(inf_x, net, train_options)
    else:
        return net(inf_x)


def main():
    args = parse_args()

    # --- Validate paired config / checkpoint counts ---
    if len(args.config) != len(args.checkpoint):
        raise ValueError(
            f'--config and --checkpoint must have the same number of arguments '
            f'(got {len(args.config)} config(s) and {len(args.checkpoint)} checkpoint(s)).')

    n_models = len(args.config)

    # --- 1. Device ---
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f'Using GPU: cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')
        print('GPU not available, using CPU.')

    # --- 2. Load all configs and build all models ---
    all_train_options = []
    nets = []
    for i, (config_path, ckpt_path) in enumerate(zip(args.config, args.checkpoint)):
        print(f'\n[Model {i+1}/{n_models}] Config: {config_path}  Checkpoint: {ckpt_path}')
        cfg_i = Config.fromfile(str(config_path))
        opts_i = copy.deepcopy(cfg_i.train_options)
        opts_i = get_variable_options(opts_i)
        net_i = get_model(opts_i, device)
        net_i = load_checkpoint(net_i, ckpt_path, device)
        net_i.eval()
        all_train_options.append(opts_i)
        nets.append(net_i)

    # Dataset settings come from the first config
    train_options = all_train_options[0]
    train_options['val_path'] = str(args.val_list)

    # Sanity-check that all models agree on charts and n_classes
    for i, opts_i in enumerate(all_train_options[1:], start=2):
        if opts_i['charts'] != train_options['charts']:
            raise ValueError(f'Model {i} charts {opts_i["charts"]} != model 1 charts {train_options["charts"]}')
        if opts_i['n_classes'] != train_options['n_classes']:
            raise ValueError(f'Model {i} n_classes mismatch with model 1')

    if n_models == 1:
        print('\nSingle-model evaluation.')
    else:
        print(f'\nEnsemble of {n_models} models (logit-level averaging).')

    # --- 3. Build validation loader ---
    dataloader, val_files = build_val_loader(train_options, args.val_list)
    print(f'Validation scenes: {len(val_files)}')

    # --- 4. Setup output ---
    if args.no_save:
        output_dir = None
    elif args.output_dir is not None:
        output_dir = str(args.output_dir)
    else:
        ckpt_dir = osp.dirname(str(args.checkpoint[0]))
        output_dir = osp.join(ckpt_dir, 'eval_scenes')
    if output_dir:
        mkdir_or_exist(output_dir)
        print(f'Results will be saved to: {output_dir}')

    # --- 5. Per-scene evaluation ---
    per_scene_results = {}
    global_cm = {
        chart: torch.zeros((train_options['n_classes'][chart], train_options['n_classes'][chart]),
                          dtype=torch.float32, device=device)
        for chart in train_options['charts']
    }

    print('\nRunning per-scene evaluation...')
    for inf_x, inf_y, cfv_masks, tfv_mask, scene_name, original_size in tqdm(
            iterable=dataloader, total=len(val_files), colour='green'):

        scene_id = osp.splitext(scene_name)[0]
        torch.cuda.empty_cache()

        inf_x = inf_x.to(device, non_blocking=True)
        with torch.no_grad(), torch.cuda.amp.autocast():
            # Run all models and average their logits per chart
            all_model_outputs = [_infer_single(inf_x, net_i, opts_i)
                                  for net_i, opts_i in zip(nets, all_train_options)]
            output = {
                chart: torch.stack([out_i[chart] for out_i in all_model_outputs], dim=0).mean(dim=0)
                for chart in train_options['charts']
            }

            # Decode predicted class maps
            output_class = {}
            for chart in train_options['charts']:
                output_class[chart] = class_decider(output[chart], train_options, chart).detach()

            # Upsample masks to original resolution (matching test_upload_function.py)
            tfv_mask = torch.nn.functional.interpolate(
                tfv_mask.type(torch.uint8).unsqueeze(0).unsqueeze(0),
                size=original_size, mode='nearest').squeeze().squeeze().to(torch.bool)
            if cfv_masks is not None:
                for chart in train_options['charts']:
                    masks_int = cfv_masks[chart].to(torch.uint8)
                    masks_int = torch.nn.functional.interpolate(
                        masks_int.unsqueeze(0).unsqueeze(0),
                        size=original_size, mode='nearest').squeeze().squeeze()
                    cfv_masks[chart] = torch.gt(masks_int, 0)

            # Upsample predictions and labels to original resolution
            if train_options['down_sample_scale'] != 1:
                for chart in train_options['charts']:
                    # Check if regression output (channels-last with n_classes=1)
                    if output[chart].size(3) == 1:
                        output[chart] = output[chart].permute(0, 3, 1, 2)
                        output[chart] = torch.nn.functional.interpolate(
                            output[chart], size=original_size, mode='nearest')
                        output[chart] = output[chart].permute(0, 2, 3, 1)
                    else:
                        output[chart] = output_class[chart].unsqueeze(0).unsqueeze(0).float()
                        output[chart] = torch.nn.functional.interpolate(
                            output[chart], size=original_size, mode='nearest').squeeze().long()
                        output_class[chart] = output[chart]

                    if inf_y is not None:
                        y_t = _ensure_tensor(inf_y[chart]).unsqueeze(0).unsqueeze(0).float()
                        y_t = torch.nn.functional.interpolate(
                            y_t, size=original_size, mode='nearest').squeeze()
                        inf_y[chart] = y_t.cpu().numpy()

            # --- Compute scene-level metrics ---
            scene_result = compute_scene_metrics(output_class, inf_y, cfv_masks, train_options)
            per_scene_results[scene_id] = scene_result
            print_scene_result(scene_id, scene_result, train_options)

            # --- Incrementally update global confusion matrix ---
            for chart in train_options['charts']:
                pred_1d = output_class[chart].flatten()
                true_1d = _ensure_tensor(inf_y[chart]).flatten().long()
                mask_1d = cfv_masks[chart].flatten()
                global_cm[chart] = _update_confusion_matrix(
                    global_cm[chart],
                    pred_1d[~mask_1d].to(device),
                    true_1d[~mask_1d].to(device),
                    train_options['n_classes'][chart])

    # --- 7. Global aggregation (derived from confusion matrices, no large tensors) ---
    print('\nComputing global metrics...')
    global_result = derive_metrics_from_cm(global_cm, train_options)
    print_global_result(global_result, train_options)

    # --- 8. Summary table ---
    target_chart = train_options.get('target_chart', train_options['charts'][0])
    print(f"\n{'='*100}")
    print(f"  PER-SCENE SUMMARY")
    print(f"{'='*100}")
    header = f"  {'Scene':<45s} {'mIoU':>8s} {'F1(w)':>8s} {'OA':>8s}"
    print(header)
    print("  " + "-" * len(header))
    for scene_id in sorted(per_scene_results.keys()):
        r = per_scene_results[scene_id]
        short_name = scene_id[:44] if len(scene_id) > 44 else scene_id
        miou_v = r['miou'][target_chart]
        f1_v   = r['f1_weighted'][target_chart]
        oa_v   = r['oa'][target_chart]
        import math
        if any(isinstance(v, float) and math.isnan(v) for v in [miou_v, f1_v, oa_v]):
            print(f"  {short_name:<45s}      N/A      N/A      N/A  [no valid pixels]")
        else:
            print(f"  {short_name:<45s} "
                  f"{miou_v:8.4f} "
                  f"{f1_v:8.4f} "
                  f"{oa_v:8.4f}")

    g = global_result
    print("  " + "-" * len(header))
    print(f"  {'<<< GLOBAL (all scenes) >>>':<45s} "
          f"{g['miou'][target_chart]:8.4f} "
          f"{g['f1_weighted'][target_chart]:8.4f} "
          f"{g['oa'][target_chart]:8.4f}")

    # --- 9. Save ---
    if output_dir:
        def _to_native(obj):
            if isinstance(obj, dict):
                return {k: _to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_to_native(v) for v in obj]
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        output_data = _to_native({
            'config': [str(c) for c in args.config],
            'checkpoint': [str(c) for c in args.checkpoint],
            'val_list': str(args.val_list),
            'num_scenes': len(val_files),
            'target_chart': target_chart,
            'per_scene': per_scene_results,
            'global': global_result,
        })

        save_path = osp.join(output_dir, 'eval_results.json')
        with open(save_path, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f'\nResults saved to: {save_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
