import argparse
import copy
import csv
import json
import random
import os
import os.path as osp
import re
import shutil
import sys
import tempfile
from icecream import ic
import pathlib
import warnings
import torchmetrics


class _Tee:
    """Duplicate stdout to a log file so all print() calls are recorded."""
    def __init__(self, log_path):
        self._file = open(log_path, 'a', encoding='utf-8')
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()

warnings.filterwarnings("ignore")

import numpy as np
import torch
from mmcv import Config, mkdir_or_exist
from tqdm import tqdm  # Progress bar

import wandb
# Functions to calculate metrics and show the relevant chart colorbar.
from src.functions import compute_metrics, save_best_model, load_model, slide_inference, \
    batched_slide_inference, fast_tiled_val_inference, class_decider, \
    create_train_validation_and_test_scene_list, \
    get_scheduler, get_optimizer, get_loss, get_model, compute_classwise_IoU, compute_mIoU, \
    compute_classwise_precision_recall, compute_normalized_confusion_matrix, compute_class_proportions, \
    compute_ordinal_cost_metric, compute_navigation_risk_metrics, select_cost_matrix, COST_MATRICES, \
    plot_cost_weighted_confusion_matrix

# Custom dataloaders for regular training and validation.
from src.loaders import get_variable_options, AI4ArcticChallengeDataset, AI4ArcticChallengeTestDataset, preload_scene_cache
#  get_variable_options

# -- Built-in modules -- #
from src.utils import colour_str
from src.test_upload_function import test


def parse_args():
    parser = argparse.ArgumentParser(description='Train Default U-NET segmentor')

    # Mandatory arguments
    parser.add_argument('config', type=pathlib.Path, help='train config file path',)
    parser.add_argument('--wandb-project', required=True, help='Name of wandb project')
    parser.add_argument('--wandb-name', default='MyDS', help='Name of wandb run (default: MyDS)')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--seed', default=None,
                        help='the seed to use, if not provided, seed from config file will be taken')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume-from', type=pathlib.Path, default=None, # 恢复训练
                       help='Resume Training from checkpoint, it will use the \
                        optimizer and schduler defined on checkpoint')
    group.add_argument('--finetune-from', type=pathlib.Path, default=None,  # 微调/迁移学习
                       help='Start new tranining using the weights from checkpoitn')

    args = parser.parse_args()

    return args


def save_epoch_sod_distribution(save_path, epoch, class_counts, mask_count, chart='SOD'):
    """Append one row of epoch-level label pixel distribution to CSV."""
    prefix = chart.lower()
    fieldnames = ['epoch'] + [f'{prefix}_{c}' for c in range(len(class_counts))] + [f'{prefix}_mask']
    write_header = not osp.exists(save_path)

    with open(save_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        row = {'epoch': epoch, f'{prefix}_mask': int(mask_count)}
        for c, count in enumerate(class_counts):
            row[f'{prefix}_{c}'] = int(count)
        writer.writerow(row)


def train(cfg, train_options, net, device, dataloader_train, dataloader_val, optimizer, scheduler, start_epoch=0):
    '''
    Trains the model.

    '''
    best_primary_miou = -np.Inf  # Best validation mIoU for the target chart.
    best_ordinal_score = -np.Inf  # Best ordinal score (方案7使用)
    best_nrs_score = -np.Inf      # Best Navigation Risk Score
    best_mean_f1_score = -np.Inf  # Best simple mean F1 (classwise average, 0-1)
    best_nrs_f1_score = -np.Inf   # Best simple mean F1 subject to NRS > baseline
    model_path = None              # Set when first best checkpoint is saved
    target_chart = train_options.get('target_chart', 'SOD')
    model_save_criterion = train_options.get('model_save_criterion', 'miou')  # 'miou' | 'ordinal_score' | 'nrs' | 'mean_f1' | 'nrs_f1'

    # Early stopping
    early_stop_patience = train_options.get('early_stop_patience', 10)
    early_stop_counter = 0

    # 验证时使用原始（未编译）模型，避免动态形状触发 recompile 警告
    net_val = net._orig_mod if train_options.get('compile_model') and hasattr(net, '_orig_mod') else net

    # 训练集 IoU（仅对有效任务）
    primary_idx = train_options['charts'].index(target_chart)
    compute_train_primary_iou = (train_options['task_weights'][primary_idx] != 0)
    primary_n_classes = train_options['n_classes'][target_chart]
    if compute_train_primary_iou:
        train_iou_metric = torchmetrics.classification.MulticlassJaccardIndex(
            num_classes=primary_n_classes, average='none', ignore_index=255).to(device)

    loss_ce_functions = {chart: get_loss(train_options['chart_loss'][chart]['type'], chart=chart, **train_options['chart_loss'][chart]).to(device)
                         for chart in train_options['charts']}

    print('Training...')
    # -- Training Loop -- #
    for epoch in tqdm(iterable=range(start_epoch, train_options['epochs'])):
        # gc.collect()  # Collect garbage to free memory.
        train_loss_sum = torch.tensor([0.])  # To sum the training batch losses during the epoch.
        cross_entropy_loss_sum = torch.tensor([0.])  # To sum the training cross entropy batch losses during the epoch.
        # To sum the training edge consistency batch losses during the epoch.
        val_loss_sum = torch.tensor([0.])  # To sum the validation batch losses during the epoch.
        # To sum the validation cross entropy batch losses during the epoch.
        val_cross_entropy_loss_sum = torch.tensor([0.])

        # Aggregate target chart label distribution over all training pixels in this epoch.
        primary_num_classes = train_options['n_classes'][target_chart]
        primary_epoch_class_counts = np.zeros(primary_num_classes, dtype=np.int64)
        primary_epoch_mask_count = 0
        # Online confusion matrix for train-set diagnostics.
        train_primary_confmat = torch.zeros((primary_num_classes, primary_num_classes), dtype=torch.float32, device=device)

        net.train()  # Set network to evaluation mode.


        #===============================================================#
        #============================训练循环============================#
        #===============================================================#
        for i, (batch_x, batch_y) in enumerate(tqdm(iterable=dataloader_train, total=train_options['epoch_len'],
                                                    colour='red')):
            # torch.cuda.empty_cache()  # Empties the GPU cache freeing up memory.
            train_loss_batch = torch.tensor([0.]).to(device)  # Reset from previous batch.
            cross_entropy_loss = torch.tensor([0.]).to(device)
            # - Transfer to device.
            batch_x = batch_x.to(device, non_blocking=True)

            # Epoch-level SOD pixel stats are computed in the main process, so
            # this works with num_workers > 0.
            if target_chart in batch_y:
                primary_batch = batch_y[target_chart]
                for c in range(primary_num_classes):
                    primary_epoch_class_counts[c] += int((primary_batch == c).sum().item())
                primary_epoch_mask_count += int((primary_batch == 255).sum().item())

            # - Mixed precision training. (Saving memory)
            with torch.cuda.amp.autocast():

                #=========模型前向传播
                output = net(batch_x)
                # breakpoint()

                #==========计算损失
                for chart, weight in zip(train_options['charts'], train_options['task_weights']):
                    cross_entropy_loss += weight * loss_ce_functions[chart](
                        output[chart], batch_y[chart].long().to(device))
            train_loss_batch = cross_entropy_loss

            # - Reset gradients from previous pass.
            optimizer.zero_grad()
            # - Backward pass.
            train_loss_batch.backward()
            # - Optimizer step
            optimizer.step()
            # - Scheduler step（ReduceLROnPlateau 改为 per-epoch，其余类型仍 per-batch）
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

            # - Add batch loss.
            train_loss_sum += train_loss_batch.detach().item()
            cross_entropy_loss_sum += cross_entropy_loss.detach().item()

            # 更新训练集 IoU 统计
            if compute_train_primary_iou:
                with torch.no_grad():
                    primary_pred_train = output[target_chart].detach().float().argmax(dim=1)
                    primary_true_train = batch_y[target_chart].to(device).long()
                    train_iou_metric.update(primary_pred_train, primary_true_train)

                    # Update confusion matrix incrementally to avoid storing full-epoch pixels.
                    valid = ((primary_true_train >= 0) & (primary_true_train < primary_num_classes) &
                             (primary_pred_train >= 0) & (primary_pred_train < primary_num_classes))
                    if valid.any():
                        t = primary_true_train[valid].view(-1)
                        p = primary_pred_train[valid].view(-1)
                        flat_idx = t * primary_num_classes + p
                        binc = torch.bincount(flat_idx, minlength=primary_num_classes * primary_num_classes).float()
                        train_primary_confmat += binc.view(primary_num_classes, primary_num_classes)

        #===========一个Epoch结束，计算平均损失
        train_loss_epoch = torch.true_divide(train_loss_sum, i + 1).detach().item()
        cross_entropy_epoch = torch.true_divide(cross_entropy_loss_sum, i + 1).detach().item()

        # 保存本epoch的像素分布统计（支持 num_workers > 0）
        patch_log_path = osp.join(cfg.work_dir, 'patch_sampling_log.csv')
        save_epoch_sod_distribution(
            patch_log_path,
            epoch,
            primary_epoch_class_counts,
            primary_epoch_mask_count,
            chart=target_chart)

        # Dynamic FocalLoss weight update (inverse frequency from actual pixel distribution)
        _dyn_enabled = train_options.get('dynamic_loss_weight', False)
        _dyn_warmup  = train_options.get('dynamic_weight_warmup', 5)
        if (_dyn_enabled
                and epoch >= _dyn_warmup
                and primary_epoch_class_counts.sum() > 0
                and target_chart in loss_ce_functions):
            _counts = primary_epoch_class_counts.astype(np.float64) + 1.0  # +1 avoid /0
            _freq   = _counts / _counts.sum()
            _raw_w  = 1.0 / _freq
            _raw_w  = _raw_w / _raw_w.mean()          # normalise to mean=1
            _max_w  = float(train_options.get('dynamic_weight_max', 5.0))
            _raw_w  = np.clip(_raw_w, 1.0 / _max_w, _max_w)
            _raw_w  = _raw_w / _raw_w.mean()          # re-normalise after clip
            _ema    = float(train_options.get('dynamic_weight_ema', 0.7))
            _loss_fn = loss_ce_functions[target_chart]
            if hasattr(_loss_fn, 'weight') and _loss_fn.weight is not None:
                _old_w = _loss_fn.weight.detach().cpu().numpy().astype(np.float64)
                _new_w = _ema * _old_w + (1.0 - _ema) * _raw_w
                _loss_fn.weight.data.copy_(torch.FloatTensor(_new_w).to(device))
                print(f"  [DynW] {target_chart} loss weights: {[f'{w:.3f}' for w in _new_w]}")
                wandb.log(
                    {f"{target_chart}/DynLossWeight_cls{c}": float(w)
                     for c, w in enumerate(_new_w)},
                    step=epoch)
            else:
                print(f"  [DynW] {target_chart} loss ({type(_loss_fn).__name__}) has no per-class weight, skipping.")

        # 计算并打印/记录训练集 mIoU
        if compute_train_primary_iou:
            train_iou_per_class = train_iou_metric.compute()  # shape: (n_classes,)
            train_miou = train_iou_per_class.mean()
            train_iou_metric.reset()

            # Derive train-set precision/recall/proportions/confusion-matrix from online CM.
            train_tp = torch.diag(train_primary_confmat)
            train_pred_sum = train_primary_confmat.sum(dim=0)
            train_true_sum = train_primary_confmat.sum(dim=1)
            train_precision_per_class = torch.where(
                train_pred_sum > 0, train_tp / train_pred_sum, torch.zeros_like(train_tp))
            train_recall_per_class = torch.where(
                train_true_sum > 0, train_tp / train_true_sum, torch.zeros_like(train_tp))
            train_row_sum = train_primary_confmat.sum(dim=1, keepdim=True)
            train_norm_confmat = torch.where(
                train_row_sum > 0, train_primary_confmat / train_row_sum, torch.zeros_like(train_primary_confmat))
            train_total = train_true_sum.sum().clamp_min(1.0)
            train_true_props = train_true_sum / train_total
            train_pred_props = train_pred_sum / train_total

            print(f"Train {target_chart} mIoU: {train_miou:.4f}")
            print(f"Train {target_chart} IoU per class: {train_iou_per_class}")
            print(f"Train {target_chart} Precision per class: {train_precision_per_class}")
            print(f"Train {target_chart} Recall per class: {train_recall_per_class}")
            print(f"Train {target_chart} True class proportion: {train_true_props}")
            print(f"Train {target_chart} Pred class proportion: {train_pred_props}")
            print(f"Train {target_chart} Normalized Confusion Matrix (row=true, col=pred):")
            print(train_norm_confmat)

            wandb.log({f"Train {target_chart} mIoU": train_miou.item()}, step=epoch)
            for c, iou_c in enumerate(train_iou_per_class):
                wandb.log({f"Train {target_chart}/IoU Class {c}": iou_c.item()}, step=epoch)
                wandb.log({f"Train {target_chart}/Precision Class {c}": train_precision_per_class[c].item()}, step=epoch)
                wandb.log({f"Train {target_chart}/Recall Class {c}": train_recall_per_class[c].item()}, step=epoch)
                wandb.log({f"Train {target_chart}/True Proportion Class {c}": train_true_props[c].item()}, step=epoch)
                wandb.log({f"Train {target_chart}/Pred Proportion Class {c}": train_pred_props[c].item()}, step=epoch)
            for r in range(train_norm_confmat.shape[0]):
                for c in range(train_norm_confmat.shape[1]):
                    wandb.log({f"Train {target_chart}/ConfMatNorm r{r}c{c}": train_norm_confmat[r, c].item()}, step=epoch)
        else:
            train_iou_per_class = None

        #===============================================================#
        #===========================验证循环=============================#
        #===============================================================#
        val_freq = train_options.get('val_freq', 1)
        do_validate = (epoch % val_freq == 0) or (epoch == train_options['epochs'] - 1)

        if not do_validate:
            # 跳过验证，只记录训练指标
            wandb.log({"Train Epoch Loss": train_loss_epoch,
                       "Train Cross Entropy Epoch Loss": cross_entropy_epoch,
                       "Learning Rate": optimizer.param_groups[0]["lr"]}, step=epoch)
            print(f"Train Epoch Loss: {train_loss_epoch:.3f}")
            continue

        # - Stores the output and the reference pixels to calculate the scores after inference on all the scenes.
        torch.cuda.empty_cache()  # 释放训练残留的缓存显存，为滑窗推理腾出空间
        outputs_flat = {chart: torch.Tensor().to(device) for chart in train_options['charts']}
        inf_ys_flat = {chart: torch.Tensor().to(device) for chart in train_options['charts']}
        net.eval()  # Set network to evaluation mode.
        print('Validating...')
        # - Loops though scenes in queue.
        for i, (inf_x, inf_y, cfv_masks, tfv_mask, name, original_size) in enumerate(tqdm(iterable=dataloader_val,
                                                                            total=len(train_options['validate_list']),
                                                                            colour='green')):
            torch.cuda.empty_cache()
            # Reset from previous batch.
            # train fill value mask
            # tfv_mask = (inf_x.squeeze()[0, :, :] == train_options['train_fill_value']).squeeze()
            val_loss_batch = torch.tensor([0.]).to(device)
            val_cross_entropy_loss = torch.tensor([0.]).to(device)
            # - Ensures that no gradients are calculated, which otherwise take up a lot of space on the GPU.
            with torch.no_grad(), torch.cuda.amp.autocast():
                inf_x = inf_x.to(device, non_blocking=True)

                #==================推理：Swin用批量滑窗，UNet用批量非重叠tile（更快且不OOM）
                if train_options['model_selection'] == 'swin':
                    output = batched_slide_inference(inf_x, net_val, train_options, 'val')
                elif (inf_x.shape[2] > train_options['patch_size'] or
                      inf_x.shape[3] > train_options['patch_size']):
                    output = fast_tiled_val_inference(inf_x, net_val, train_options)
                else:
                    output = net_val(inf_x)

                for chart, weight in zip(train_options['charts'], train_options['task_weights']):

                    val_cross_entropy_loss += weight * loss_ce_functions[chart](output[chart],
                                                                                inf_y[chart].unsqueeze(0).long().to(device))

            val_loss_batch = val_cross_entropy_loss

            # - Final output layer, and storing of non masked pixels.
            for chart in train_options['charts']:
                output[chart] = class_decider(output[chart], train_options, chart)
                # output[chart] = torch.argmax(
                #     output[chart], dim=1).squeeze()
                outputs_flat[chart] = torch.cat((outputs_flat[chart], output[chart][~cfv_masks[chart]]))
                inf_ys_flat[chart] = torch.cat((inf_ys_flat[chart], inf_y[chart]
                                                [~cfv_masks[chart]].to(device, non_blocking=True)))
            # - Add batch loss.
            val_loss_sum += val_loss_batch.detach().item()
            val_cross_entropy_loss_sum += val_cross_entropy_loss.detach().item()

        #====================验证结束后，计算综合指标
        val_loss_epoch = torch.true_divide(val_loss_sum, i + 1).detach().item()
        val_cross_entropy_epoch = torch.true_divide(val_cross_entropy_loss_sum, i + 1).detach().item()

        # - Compute the relevant scores.
        print('Computing Metrics on Val dataset')
        combined_score, scores = compute_metrics(true=inf_ys_flat, pred=outputs_flat, charts=train_options['charts'],
                                                 metrics=train_options['chart_metric'], num_classes=train_options['n_classes'])

        # 始终计算 mIoU，用于按目标任务 mIoU 保存最佳模型。
        miou_scores = compute_mIoU(true=inf_ys_flat, pred=outputs_flat,
                                   charts=train_options['charts'], num_classes=train_options['n_classes'])
        primary_miou = miou_scores[target_chart]

        # 有序代价指标（方案1/4/7）：临近错误-1分，远处错误-2分
        _ordinal_cfg = train_options.get('ordinal_metric', {})
        if _ordinal_cfg.get('enabled', False):
            ordinal_raw_costs, ordinal_norm_scores = compute_ordinal_cost_metric(
                true=inf_ys_flat, pred=outputs_flat,
                charts=train_options['charts'], num_classes=train_options['n_classes'],
                cost_per_distance=_ordinal_cfg.get('cost_per_distance', None),
            )
        else:
            ordinal_raw_costs, ordinal_norm_scores = None, None

        # NRS/OVR/UR — asymmetric navigation-risk metrics (文档 §4)
        nrs_scores, ovr_scores, ur_scores = compute_navigation_risk_metrics(
            true=inf_ys_flat, pred=outputs_flat,
            charts=train_options['charts'], num_classes=train_options['n_classes'],
        )

        # Reference NRS using the fixed 'current' matrix for cross-experiment comparison.
        # When cost_matrix_variant='current' (default), nrs_ref_scores == nrs_scores.
        _cost_variant = train_options.get('cost_matrix_variant', 'current')
        if _cost_variant != 'current':
            nrs_ref_scores, _, _ = compute_navigation_risk_metrics(
                true=inf_ys_flat, pred=outputs_flat,
                charts=train_options['charts'], num_classes=train_options['n_classes'],
                cost_matrix=COST_MATRICES['current'],
            )
        else:
            nrs_ref_scores = nrs_scores

        # 统一计算 per-class IoU，便于做 train-val IoU gap 分析
        classwise_iou_scores = compute_classwise_IoU(true=inf_ys_flat, pred=outputs_flat,
                                                     charts=train_options['charts'], num_classes=train_options['n_classes'])

        # 计算全类别 Precision/Recall、归一化混淆矩阵、预测占比 vs 真实占比
        classwise_precision_scores, classwise_recall_scores = compute_classwise_precision_recall(
            true=inf_ys_flat, pred=outputs_flat,
            charts=train_options['charts'], num_classes=train_options['n_classes'])
        normalized_conf_mats = compute_normalized_confusion_matrix(
            true=inf_ys_flat, pred=outputs_flat,
            charts=train_options['charts'], num_classes=train_options['n_classes'])
        true_class_props, pred_class_props = compute_class_proportions(
            true=inf_ys_flat, pred=outputs_flat,
            charts=train_options['charts'], num_classes=train_options['n_classes'])

        if train_options['compute_classwise_f1score']:
            from src.functions import compute_classwise_f1score, compute_overall_accuracy

            # 计算每类的 F1
            classwise_scores = compute_classwise_f1score(true=inf_ys_flat, pred=outputs_flat,
                                                         charts=train_options['charts'], num_classes=train_options['n_classes'])

            # 计算 OA
            oa_scores = compute_overall_accuracy(true=inf_ys_flat, pred=outputs_flat, charts=train_options['charts'])

        # 立即释放验证集大 tensor，避免其在下一 epoch 训练期间持续占用内存
        del outputs_flat, inf_ys_flat
        import gc; gc.collect()
        torch.cuda.empty_cache()

        print("")
        print(f"Epoch {epoch} score:")

        for chart in train_options['charts']:
            # 跳过 task_weight=0 的任务，避免无意义的指标日志
            if train_options['chart_metric'][chart]['weight'] == 0:
                continue

            print(f"{chart} {train_options['chart_metric'][chart]['func'].__name__}: {scores[chart]}%")

            # Log in wandb the SIC r2_metric, SOD f1_metric and FLOE f1_metric
            wandb.log({f"{chart} {train_options['chart_metric'][chart]['func'].__name__}": scores[chart]}, step=epoch)

            #===============计算标签各类别的F1分数
            if train_options['compute_classwise_f1score']:
                for index, class_score in enumerate(classwise_scores[chart]):
                    wandb.log({f"{chart}/Class: {index}": class_score.item()}, step=epoch)
                print(f"{chart} F1 score:", classwise_scores[chart])

                # 打印或记录 OA
                print(f"{chart} OA:", oa_scores[chart])
                wandb.log({f"{chart}/OA": oa_scores[chart]}, step=epoch)

                # 打印/记录 mIoU
                print(f"{chart} mIoU:", miou_scores[chart])
                wandb.log({f"{chart}/mIoU": miou_scores[chart]}, step=epoch)

                # 打印/记录 per-class IoU
                print(f"{chart} IoU per class:", classwise_iou_scores[chart])
                for c, iou_c in enumerate(classwise_iou_scores[chart]):
                    wandb.log({f"{chart}/IoU Class {c}": iou_c.item()}, step=epoch)

            # 打印/记录有序代价指标（方案1/4/7）
            if ordinal_raw_costs is not None and chart in ordinal_raw_costs:
                print(f"{chart} Ordinal Mean Cost (lower=better): {ordinal_raw_costs[chart]:.4f}")
                print(f"{chart} Ordinal Score (higher=better):    {ordinal_norm_scores[chart]:.4f}")
                wandb.log({f"{chart}/OrdinalMeanCost":  ordinal_raw_costs[chart].item(),
                           f"{chart}/OrdinalScore":     ordinal_norm_scores[chart].item()}, step=epoch)

            # 打印/记录 NRS / OVR / UR（文档 §4）
            if chart in nrs_scores:
                _nrs = nrs_scores[chart].item()
                _ovr = ovr_scores[chart].item()
                _ur  = ur_scores[chart].item()
                print(f"{chart} NRS (higher=better): {_nrs:.4f}  "
                      f"OVR (lower=better): {_ovr:.4f}  "
                      f"UR (lower=better):  {_ur:.4f}")
                wandb.log({f"{chart}/NRS": _nrs,
                           f"{chart}/OVR": _ovr,
                           f"{chart}/UR":  _ur}, step=epoch)

                # Reference NRS (current matrix) — comparable across cost_matrix_variant experiments
                if chart in nrs_ref_scores:
                    _nrs_ref = nrs_ref_scores[chart].item()
                    if _cost_variant != 'current':
                        print(f"{chart} NRS_ref (current matrix, higher=better): {_nrs_ref:.4f}")
                    wandb.log({f"{chart}/NRS_ref": _nrs_ref}, step=epoch)

            # 打印/记录 per-class Precision
            print(f"{chart} Precision per class:", classwise_precision_scores[chart])
            for c, prec_c in enumerate(classwise_precision_scores[chart]):
                wandb.log({f"{chart}/Precision Class {c}": prec_c.item()}, step=epoch)

            # 打印/记录 per-class Recall
            print(f"{chart} Recall per class:", classwise_recall_scores[chart])
            for c, rec_c in enumerate(classwise_recall_scores[chart]):
                wandb.log({f"{chart}/Recall Class {c}": rec_c.item()}, step=epoch)

            # 打印/记录 预测占比 vs 真实占比
            print(f"{chart} True class proportion:", true_class_props[chart])
            print(f"{chart} Pred class proportion:", pred_class_props[chart])
            for c, p_true in enumerate(true_class_props[chart]):
                wandb.log({f"{chart}/True Proportion Class {c}": p_true.item()}, step=epoch)
            for c, p_pred in enumerate(pred_class_props[chart]):
                wandb.log({f"{chart}/Pred Proportion Class {c}": p_pred.item()}, step=epoch)

            # 打印/记录 归一化混淆矩阵（按真实类别行归一化）
            cm_norm = normalized_conf_mats[chart]
            print(f"{chart} Normalized Confusion Matrix (row=true, col=pred):")
            print(cm_norm)
            for r in range(cm_norm.shape[0]):
                for c in range(cm_norm.shape[1]):
                    wandb.log({f"{chart}/ConfMatNorm r{r}c{c}": cm_norm[r, c].item()}, step=epoch)

        # 代价加权混淆矩阵（绘图 + wandb 上传）
        if train_options.get('plot_cost_weighted_confusion_matrix', False):
            cwcm_figs = plot_cost_weighted_confusion_matrix(
                norm_cms=normalized_conf_mats,
                charts=train_options['charts'],
                num_classes=train_options['n_classes'],
                save_dir=osp.join(cfg.work_dir, 'cost_cm'),
                epoch=epoch,
            )
            for _chart, _fig in cwcm_figs.items():
                wandb.log({f"{_chart}/CostWeightedCM": wandb.Image(_fig)}, step=epoch)
                import matplotlib.pyplot as _plt
                _plt.close(_fig)

        # 计算 Train-Val IoU gap
        if compute_train_primary_iou and train_iou_per_class is not None and target_chart in classwise_iou_scores:
            train_val_iou_gap = train_iou_per_class - classwise_iou_scores[target_chart]
            print(f"{target_chart} Train-Val IoU gap per class:", train_val_iou_gap)
            for c, gap_c in enumerate(train_val_iou_gap):
                wandb.log({f"{target_chart}/Train-Val IoU Gap Class {c}": gap_c.item()}, step=epoch)

        # ReduceLROnPlateau：用目标任务 mIoU 驱动 lr 调整，与模型保存标准保持一致
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_before = optimizer.param_groups[0]['lr']
            scheduler.step(primary_miou)
            lr_after = optimizer.param_groups[0]['lr']
            if lr_after < lr_before:
                print(f"ReduceLROnPlateau: lr {lr_before:.2e} → {lr_after:.2e}")

        print(f"Combined score: {combined_score}%")
        print(f"Train Epoch Loss: {train_loss_epoch:.3f}")
        print(f"Train Cross Entropy Epoch Loss: {cross_entropy_epoch:.3f}")
        print(f"Validation Epoch Loss: {val_loss_epoch:.3f}")
        print(f"Validation Cross Entropy Epoch Loss: {val_cross_entropy_epoch:.3f}")
        print(f"Validation {target_chart} mIoU: {primary_miou.item():.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        # Log combine score and epoch loss to wandb
        wandb.log({"Combined score": combined_score,    # 综合得分（验证集）
                   f"Validation {target_chart} mIoU": primary_miou.item(),
                   "Train Epoch Loss": train_loss_epoch,
                   "Train Cross Entropy Epoch Loss": cross_entropy_epoch,
                   "Validation Epoch Loss": val_loss_epoch,
                   "Validation Cross Entropy Epoch Loss": val_cross_entropy_epoch,
                   "Learning Rate": optimizer.param_groups[0]["lr"]}, step=epoch)

        # -- Attention Residual alpha 监控（仅 Swin 模型）--
        # alpha 零初始化，若训练中逐渐增大说明模型学到了跨层注意力残差的价值。
        # 若始终为 0 且梯度为 0，说明残差链路存在问题（如被意外截断）。
        # 注意：alpha 变化≠效果好，最终判断依据仍是验证集 F1。
        if (train_options['model_selection'] == 'swin'
                and hasattr(net_val, 'layers')
                and len(net_val.layers) > 2):
            stage2_alphas = torch.stack([
                blk.attn.alpha.detach().cpu()
                for blk in net_val.layers[2].blocks
            ])  # [num_blocks=6, num_heads=12]
            alpha_mean = stage2_alphas.mean().item()
            alpha_max  = stage2_alphas.max().item()
            # 每个 block 在所有 head 上的均值（偶数=W-MSA，奇数=SW-MSA）
            alpha_per_block = stage2_alphas.mean(dim=1).tolist()
            print(f"[AttnResidual] Stage2 alpha | "
                  f"mean={alpha_mean:.4f}  max={alpha_max:.4f}  "
                  f"per_block(W/SW alternating)={[f'{v:.4f}' for v in alpha_per_block]}")
            wandb.log({
                "AttnResidual/alpha_mean": alpha_mean,
                "AttnResidual/alpha_max":  alpha_max,
                **{f"AttnResidual/block{i}_alpha_mean": v
                   for i, v in enumerate(alpha_per_block)},
            }, step=epoch)

        # 决定本epoch是否保存最佳模型（miou / ordinal_score / nrs）
        if model_save_criterion == 'nrs':
            _current_criterion = nrs_scores.get(target_chart, torch.tensor(-np.Inf))
            _current_criterion = _current_criterion.item() if hasattr(_current_criterion, 'item') else float(_current_criterion)
            _is_best = _current_criterion > best_nrs_score
            if _is_best:
                best_nrs_score = _current_criterion
        elif model_save_criterion == 'ordinal_score' and ordinal_norm_scores is not None:
            _current_criterion = ordinal_norm_scores.get(target_chart, torch.tensor(-np.Inf))
            _current_criterion = _current_criterion.item() if hasattr(_current_criterion, 'item') else float(_current_criterion)
            _is_best = _current_criterion > best_ordinal_score
            if _is_best:
                best_ordinal_score = _current_criterion
        elif model_save_criterion == 'mean_f1':
            # Simple (macro) mean of per-class F1, 0-1 range
            _current_criterion = classwise_scores[target_chart].mean().item()
            _is_best = _current_criterion > best_mean_f1_score
            if _is_best:
                best_mean_f1_score = _current_criterion
        elif model_save_criterion == 'nrs_f1':
            # Hard constraint: NRS must exceed CE_argmax baseline; among valid epochs maximize mean F1
            _nrs = nrs_scores.get(target_chart, torch.tensor(-np.Inf))
            _nrs = _nrs.item() if hasattr(_nrs, 'item') else float(_nrs)
            _f1 = classwise_scores[target_chart].mean().item()
            NRS_BASELINE = 0.9808
            _current_criterion = _f1 if _nrs > NRS_BASELINE else -np.Inf
            _is_best = _current_criterion > best_nrs_f1_score
            if _is_best:
                best_nrs_f1_score = _current_criterion
        else:
            _current_criterion = primary_miou.item()
            _is_best = _current_criterion > best_primary_miou
            if _is_best:
                best_primary_miou = _current_criterion

        # If criterion improved, save best model and reset early-stop counter.
        if _is_best:
            early_stop_counter = 0

            # Log the best criterion value under its correct name.
            if model_save_criterion == 'nrs':
                wandb.run.summary[f"While training/Best {target_chart} NRS"] = best_nrs_score
            elif model_save_criterion == 'ordinal_score':
                wandb.run.summary[f"While training/Best {target_chart} OrdinalScore"] = best_ordinal_score
            elif model_save_criterion == 'mean_f1':
                wandb.run.summary[f"While training/Best {target_chart} MeanF1"] = best_mean_f1_score
            elif model_save_criterion == 'nrs_f1':
                wandb.run.summary[f"While training/Best {target_chart} NRS_F1"] = best_nrs_f1_score
            else:
                wandb.run.summary[f"While training/Best {target_chart} mIoU"] = best_primary_miou
            for chart in train_options['charts']:
                wandb.run.summary[f"While training/{chart} {train_options['chart_metric'][chart]['func'].__name__}"] = scores[chart]
            wandb.run.summary[f"While training/Validation {target_chart} mIoU"] = primary_miou.item()
            wandb.run.summary[f"While training/Train Epoch Loss"] = train_loss_epoch

            # Save the best model in work_dirs
            model_path = save_best_model(cfg, train_options, net, optimizer, scheduler, epoch)

            if train_options.get('wandb_upload_model', False):
                wandb.save(model_path)
        else:
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter}/{early_stop_patience}")
            if early_stop_counter >= early_stop_patience:
                if model_save_criterion == 'nrs':
                    print(f"Early stopping triggered at epoch {epoch}. Best {target_chart} NRS: {best_nrs_score:.4f}")
                elif model_save_criterion == 'ordinal_score':
                    print(f"Early stopping triggered at epoch {epoch}. Best {target_chart} OrdinalScore: {best_ordinal_score:.4f}")
                elif model_save_criterion == 'mean_f1':
                    print(f"Early stopping triggered at epoch {epoch}. Best {target_chart} MeanF1: {best_mean_f1_score:.4f}")
                elif model_save_criterion == 'nrs_f1':
                    print(f"Early stopping triggered at epoch {epoch}. Best {target_chart} NRS_F1 (NRS>{NRS_BASELINE}): {best_nrs_f1_score:.4f}")
                else:
                    print(f"Early stopping triggered at epoch {epoch}. Best {target_chart} mIoU: {best_primary_miou:.4f}")
                break

    return model_path


def create_dataloaders(train_options):
    '''
    Create train and validation dataloader based on the train and validation list inside train_options.

    '''
    # Custom dataset and dataloader.
    dataset = AI4ArcticChallengeDataset(
        files=train_options['train_list'], options=train_options, do_transform=True)

    prefetch = train_options.get('prefetch_factor', 2) if train_options['num_workers'] > 0 else None
    dataloader_train = torch.utils.data.DataLoader(
        dataset, batch_size=None, shuffle=True, num_workers=train_options['num_workers'],
        pin_memory=True, prefetch_factor=prefetch)
    # - Setup of the validation dataset/dataloader. The same is used for model testing in 'test_upload.ipynb'.

    dataset_val = AI4ArcticChallengeTestDataset(
        options=train_options, files=train_options['validate_list'], mode='train')

    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=None, num_workers=train_options['num_workers_val'], shuffle=False)

    return dataloader_train, dataloader_val
    # return dataloader_val


def _set_global_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _setup_device(train_options):
    # Get GPU resources.
    if torch.cuda.is_available():
        print(colour_str('GPU available!', 'green'))
        print('Total number of available devices: ',
              colour_str(torch.cuda.device_count(), 'orange'))

        # Check if NVIDIA V100, A100, or H100 is available for torch compile speed up
        if train_options['compile_model']:
            gpu_ok = False
            major, minor = torch.cuda.get_device_capability()
            if major >= 7:
                gpu_ok = True

            if not gpu_ok:
                warnings.warn(
                    colour_str("GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower than expected.", 'red')
                )

        # Setup device to be used
        device = torch.device(f"cuda:{train_options['gpu_id']}")
    else:
        print(colour_str('GPU not available.', 'red'))
        device = torch.device('cpu')
    print('GPU setup completed!')
    return device


def _collect_cv_folds(train_options):
    fold_dir = train_options.get('cv_fold_dir', None)
    if not fold_dir:
        return []

    fold_dir_rel = fold_dir.replace('\\', '/').rstrip('/')
    fold_dir_abs = osp.join(train_options['path_to_env'], fold_dir_rel)
    if not osp.isdir(fold_dir_abs):
        raise FileNotFoundError(f'cv_fold_dir not found: {fold_dir_abs}')

    train_pat = re.compile(r'^fold(\d+)_train\.json$')
    fold_ids = []
    for name in os.listdir(fold_dir_abs):
        m = train_pat.match(name)
        if m is not None:
            fold_ids.append(int(m.group(1)))

    fold_ids = sorted(set(fold_ids))
    if not fold_ids:
        return []

    requested_num = int(train_options.get('cv_num_folds', 0) or 0)
    if requested_num > 0:
        fold_ids = fold_ids[:requested_num]

    folds = []
    for fold_id in fold_ids:
        train_rel = f'{fold_dir_rel}/fold{fold_id}_train.json'
        val_rel = f'{fold_dir_rel}/fold{fold_id}_val.json'
        train_abs = osp.join(train_options['path_to_env'], train_rel)
        val_abs = osp.join(train_options['path_to_env'], val_rel)
        if not osp.exists(train_abs):
            raise FileNotFoundError(f'Missing fold train list: {train_abs}')
        if not osp.exists(val_abs):
            raise FileNotFoundError(f'Missing fold val list: {val_abs}')
        folds.append({'fold': fold_id, 'train_list_path': train_rel, 'val_path': val_rel})

    return folds


def _read_json_list(path):
    with open(path) as f:
        return json.loads(f.read())


def _run_training_job(cfg, args, train_options, device, fold_tag=None):
    tee = _Tee(osp.join(cfg.work_dir, 'train.log'))
    sys.stdout = tee
    try:
        _run_training_job_inner(cfg, args, train_options, device, fold_tag)
    finally:
        tee.close()


def _run_training_job_inner(cfg, args, train_options, device, fold_tag=None):
    #===========根据"model_selection"选择模型，并设置优化器和学习率调度器
    net = get_model(train_options, device)
    if train_options['compile_model']:
        net = torch.compile(net)
    optimizer = get_optimizer(train_options, net)
    scheduler = get_scheduler(train_options, optimizer)

    epoch_start = 0
    #===========恢复训练或微调模型
    # 顺序CV模式下禁止 resume/finetune（每折都是全新训练）
    if fold_tag is None:
        if args.resume_from is not None:
            print(f"\033[91m Resuming work from {args.resume_from}\033[0m")
            epoch_start = load_model(net, args.resume_from, optimizer, scheduler)
        elif args.finetune_from is not None:
            print(f"\033[91m Finetune model from {args.finetune_from}\033[0m")
            _ = load_model(net, args.finetune_from)

    #===========Wandb设置
    run_name = args.wandb_name if fold_tag is None else f"{args.wandb_name}_fold{fold_tag}"
    _wandb_dir = tempfile.gettempdir()
    if not train_options['cross_val_run']:
        wandb.init(name=run_name, project=args.wandb_project,
                   entity="liwannian-zhejiang-university", config=train_options,
                   dir=_wandb_dir)
    else:
        wandb.init(name=run_name, group=osp.splitext(osp.basename(args.config))[0], project=args.wandb_project,
                   entity="liwannian-zhejiang-university", config=train_options,
                   dir=_wandb_dir)

    # Define the metrics and make them such that they are not added to the summary
    wandb.define_metric("Train Epoch Loss", summary="none")     # 训练集上的总损失
    wandb.define_metric("Train Cross Entropy Epoch Loss", summary="none") # 训练集上的交叉熵损失
    wandb.define_metric("Train Water Consistency Epoch Loss", summary="none") # 训练集上的水体一致性损失
    wandb.define_metric("Validation Epoch Loss", summary="none") # 验证集上的总损失
    wandb.define_metric("Validation Cross Entropy Epoch Loss", summary="none") # 验证集上的交叉熵损失
    wandb.define_metric("Validation Water Consistency Epoch Loss", summary="none") # 验证集上的水体一致性损失
    wandb.define_metric("Combined score", summary="none") # 综合得分
    wandb.define_metric("SIC r2_metric", summary="none") # SIC的R2指标
    wandb.define_metric("SOD f1_metric", summary="none") # SOD的F1指标
    wandb.define_metric("FLOE f1_metric", summary="none") # FLOE的F1指标
    wandb.define_metric(f"Validation {train_options.get('target_chart', 'SOD')} mIoU", summary="none") # 验证集目标任务的mIoU
    wandb.define_metric("Water Consistency Accuarcy", summary="none") # 水体一致性准确率
    wandb.define_metric("Learning Rate", summary="none") # 学习率
    wandb.save(str(args.config))
    print(colour_str('Save Config File', 'green'))

    # ===========创建数据加载器
    create_train_validation_and_test_scene_list(train_options)
    dataloader_train, dataloader_val = create_dataloaders(train_options) # dataloader_val 用于训练过程中验证数据

    wandb.config['validate_list'] = train_options['validate_list']
    print('Data setup complete.')

    print('-----------------------------------')
    print('Starting Training')
    print('-----------------------------------')
    if args.resume_from is not None and fold_tag is None:
        checkpoint_path = train(cfg, train_options, net, device, dataloader_train, dataloader_val, optimizer,
                                scheduler, epoch_start)
    else:
        checkpoint_path = train(cfg, train_options, net, device, dataloader_train, dataloader_val, optimizer,
                                scheduler)

    print('-----------------------------------')
    print('Training Complete')
    print('-----------------------------------')

    print('-----------------------------------')
    print('Staring Validation with best model')
    print('-----------------------------------')

    # 释放训练占用的 GPU 内存，避免最终验证时 OOM
    del dataloader_train, dataloader_val
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # this is for valset 1 visualization along with gt
    # test('val', net, checkpoint_path, device, cfg.deepcopy(), train_options['validate_list'], 'Cross Validation')

    print('-----------------------------------')
    print('Completed validation')
    print('-----------------------------------')

    print('-----------------------------------')
    print('Starting testing with best model')
    print('-----------------------------------')

    # this is for test path along with gt after the gt has been released
    # test('test', net, checkpoint_path, device, cfg.deepcopy(), train_options['test_list'], 'Test', train_options['test_list_reference'])

    print('-----------------------------------')
    print('Completed testing')
    print('-----------------------------------')

    # finish the wandb run
    wandb.finish()

    # Fold-level cleanup; global scene cache stays alive for next fold.
    del net, optimizer, scheduler
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    ic(args.config)
    cfg = Config.fromfile(args.config)
    train_options = copy.deepcopy(cfg.train_options)
    # Get options for variables, amsrenv grid, cropping and upsampling.
    train_options = get_variable_options(train_options)
    # Apply cost matrix variant before any loss or metric code runs.
    _cost_variant = train_options.get('cost_matrix_variant', 'current')
    if _cost_variant != 'current':
        select_cost_matrix(_cost_variant)
    # generate wandb run id, to be used to link the run with test_upload
    id = wandb.util.generate_id()

    #===========设置随机种子
    if train_options['seed'] != -1 or args.seed is not None:
        # set seed for everything
        if args.seed is not None:
            seed = int(args.seed)
        else:
            seed = train_options['seed']
        _set_global_seed(seed)
        print(f"Seed: {seed}")
    else:
        print("Random Seed Chosen")

    #===========确定工作目录（保存日志和权重文件）
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        base_work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        if not train_options['cross_val_run']:
            base_work_dir = osp.join('./work_dir',
                                     osp.splitext(osp.basename(args.config))[0])
        else:
            # from utils import run_names
            run_name = id
            base_work_dir = osp.join('./work_dir',
                                     osp.splitext(osp.basename(args.config))[0], run_name)
    else:
        base_work_dir = cfg.work_dir

    ic(base_work_dir)
    mkdir_or_exist(osp.abspath(base_work_dir))
    # dump config
    shutil.copy(args.config, osp.join(base_work_dir, osp.basename(args.config)))

    #=========== GPU设置
    device = _setup_device(train_options)

    # Sequential CV: one process, one-time preload, loop over folds.
    if train_options.get('sequential_cv', False):
        if args.resume_from is not None or args.finetune_from is not None:
            raise ValueError('sequential_cv mode does not support --resume-from/--finetune-from')

        folds = _collect_cv_folds(train_options)
        if not folds:
            raise ValueError('No folds discovered. Please set train_options["cv_fold_dir"] correctly.')

        print(f'Sequential CV mode enabled. Total folds: {len(folds)}')

        # Optional one-time preload for all fold train scenes.
        if train_options.get('preload_all_cv_scenes', True):
            all_train_files = []
            for fold in folds:
                train_json_abs = osp.join(train_options['path_to_env'], fold['train_list_path'])
                all_train_files.extend(_read_json_list(train_json_abs))
            print(f'Preloading unique train scenes across folds: {len(set(all_train_files))}')
            preload_scene_cache(train_options, all_train_files)

        for fold in folds:
            fold_id = fold['fold']
            print('===================================')
            print(f'Starting fold {fold_id}')
            print('===================================')

            fold_cfg = cfg.deepcopy()
            fold_cfg.work_dir = osp.join(base_work_dir, f'fold{fold_id}')
            mkdir_or_exist(osp.abspath(fold_cfg.work_dir))
            shutil.copy(args.config, osp.join(fold_cfg.work_dir, osp.basename(args.config)))

            fold_options = copy.deepcopy(train_options)
            fold_options['train_list_path'] = fold['train_list_path']
            fold_options['val_path'] = fold['val_path']

            _run_training_job(fold_cfg, args, fold_options, device, fold_tag=fold_id)

            print('===================================')
            print(f'Completed fold {fold_id}')
            print('===================================')
    else:
        cfg.work_dir = base_work_dir
        _run_training_job(cfg, args, train_options, device)


if __name__ == '__main__':
    main()
