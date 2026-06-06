#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Helping functions for 'introduction' and 'quickstart' notebooks."""

# -- File info -- #
__author__ = 'Muhammed Patel'
__contributor__ = 'Xinwwei chen, Fernando Pena Cantu,Javier Turnes, Eddie Park'
__copyright__ = ['university of waterloo']
__contact__ = ['m32patel@uwaterloo.ca', 'xinweic@uwaterloo.ca']
__version__ = '1.0.0'
__date__ = '2024-04-05'

# -- Built-in modules -- #
import os
import json
# -- Third-party modules -- #
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
# from sklearn.metrics import r2_score, f1_score
from torchmetrics.functional import r2_score, f1_score, jaccard_index  # 修改：引入 jaccard_index
from torchmetrics.functional.classification import multiclass_confusion_matrix
import segmentation_models_pytorch as smp
from tqdm import tqdm  # Progress bar
# -- Proprietary modules -- #

from src.utils import ICE_STRINGS, GROUP_NAMES
from src.unet import UNet, Sep_feat_dif_stages  # Convolutional Neural Network model
from swin_transformer import SwinTransformer  # Swin Transformer

# Asymmetric navigation-risk cost matrix C[true_class, pred_class].
# Lower triangle (underestimation) penalised 2-4x more than upper triangle
# (overestimation) to reflect navigation safety asymmetry.
# C_max = 16 (worst case: multi-year ice predicted as open water).
NAVIGATION_COST_MATRIX = torch.tensor([
    [ 0,  1,  2,  3,  4],
    [ 6,  0,  1,  2,  3],
    [ 9,  4,  0,  1,  2],
    [12,  6,  3,  0,  1],
    [16,  8,  4,  2,  0],
], dtype=torch.float32)

# ---------------------------------------------------------------------------
# Sensitivity-analysis cost matrix variants (cost_matrix_variant config key)
# All share overestimation cost = 1 per step; underestimation = α × steps.
# 'mild'     α=2: conservative asymmetry, C_max=8
# 'moderate' α=4: uniform 4:1 asymmetry,  C_max=16 (same scale as current)
# 'severe'   α=8: aggressive asymmetry,   C_max=32
# ---------------------------------------------------------------------------
_COST_MATRIX_MILD = torch.tensor([
    [ 0,  1,  2,  3,  4],
    [ 2,  0,  1,  2,  3],
    [ 4,  2,  0,  1,  2],
    [ 6,  4,  2,  0,  1],
    [ 8,  6,  4,  2,  0],
], dtype=torch.float32)

_COST_MATRIX_MODERATE = torch.tensor([
    [ 0,  1,  2,  3,  4],
    [ 4,  0,  1,  2,  3],
    [ 8,  4,  0,  1,  2],
    [12,  8,  4,  0,  1],
    [16, 12,  8,  4,  0],
], dtype=torch.float32)

_COST_MATRIX_SEVERE = torch.tensor([
    [ 0,  1,  2,  3,  4],
    [ 8,  0,  1,  2,  3],
    [16,  8,  0,  1,  2],
    [24, 16,  8,  0,  1],
    [32, 24, 16,  8,  0],
], dtype=torch.float32)

# ---------------------------------------------------------------------------
# Non-linear upper triangle variants: upper(d) = (α/2)×(d-1) + 1
# Adjacent overestimation stays cheap (cost=1); each extra step adds α/2,
# so large overestimations become progressively expensive.
# Lower triangle unchanged from moderate / severe respectively.
# 'moderate_v3'  α=4 → upper: 1,3,5,7   C_max=16
# 'severe_v3'    α=8 → upper: 1,5,9,13  C_max=32
# ---------------------------------------------------------------------------
_COST_MATRIX_MODERATE_V3 = torch.tensor([
    [ 0,  1,  3,  5,  7],
    [ 4,  0,  1,  3,  5],
    [ 8,  4,  0,  1,  3],
    [12,  8,  4,  0,  1],
    [16, 12,  8,  4,  0],
], dtype=torch.float32)

# ---------------------------------------------------------------------------
# 'severe_v3': severe lower triangle (8×d), non-linear upper triangle.
# Upper: C_upper(d) = 4d - 3  →  d=1:1, d=2:5, d=3:9, d=4:13
# Adjacent overestimation stays cheap (cost=1); large overestimation becomes
# progressively expensive. Ratio lower:upper narrows from 8:1 to 2.5:1.
# C_max = 32 (same as severe → NRS comparable).
# ---------------------------------------------------------------------------
_COST_MATRIX_SEVERE_V3 = torch.tensor([
    [ 0,  1,  5,  9, 13],
    [ 8,  0,  1,  5,  9],
    [16,  8,  0,  1,  5],
    [24, 16,  8,  0,  1],
    [32, 24, 16,  8,  0],
], dtype=torch.float32)

_COST_MATRIX_INVERTED = torch.tensor([
    [ 0,  1,  2,  3,  4],
    [ 2,  0,  1,  2,  3],
    [ 6,  4,  0,  1,  2],
    [10,  8,  4,  0,  1],
    [16, 14, 10,  6,  0],
], dtype=torch.float32)

COST_MATRICES = {
    'current':      NAVIGATION_COST_MATRIX,
    'mild':         _COST_MATRIX_MILD,
    'moderate':     _COST_MATRIX_MODERATE,
    'severe':       _COST_MATRIX_SEVERE,
    'inverted':     _COST_MATRIX_INVERTED,
    'moderate_v3':  _COST_MATRIX_MODERATE_V3,
    'severe_v3':    _COST_MATRIX_SEVERE_V3,
}


def select_cost_matrix(name: str) -> None:
    """Replace the global NAVIGATION_COST_MATRIX with a named variant.

    Must be called before any loss function is constructed or any inference /
    metric code runs.  Valid names: 'current', 'mild', 'moderate', 'severe'.
    """
    global NAVIGATION_COST_MATRIX
    if name not in COST_MATRICES:
        raise ValueError(
            f"Unknown cost_matrix_variant '{name}'. "
            f"Choose from: {list(COST_MATRICES)}"
        )
    NAVIGATION_COST_MATRIX = COST_MATRICES[name]
    print(f"[cost_matrix] variant='{name}'  C_max={NAVIGATION_COST_MATRIX.max().item():.0f}")


def chart_cbar(ax, n_classes, chart, cmap='vridis'):
    """
    Create discrete colourbar for plot with the sea ice parameter class names.

    Parameters
    ----------
    n_classes: int
        Number of classes for the chart parameter.
    chart: str
        The relevant chart.
    """
    n_labels = len(GROUP_NAMES[chart])  # number of labelled (non-mask) classes
    arranged = np.arange(0, n_labels + 1)
    cmap = plt.get_cmap(cmap, n_labels)
    # Get colour boundaries. -0.5 to center ticks for each color.
    norm = mpl.colors.BoundaryNorm(arranged - 0.5, cmap.N)
    arranged = arranged[:-1]  # Discount the boundary sentinel.
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=arranged, fraction=0.0485, pad=0.049, ax=ax)
    cbar.set_label(label=ICE_STRINGS[chart])
    cbar.set_ticklabels(list(GROUP_NAMES[chart].values()))


def compute_metrics(true, pred, charts, metrics, num_classes):
    """
    Calculates metrics for each chart and the combined score. true and pred must be 1d arrays of equal length.

    Parameters
    ----------
    true :
        ndarray, 1d contains all true pixels. Must be numpy array.
    pred :
        ndarray, 1d contains all predicted pixels. Must be numpy array.
    charts : List
        List of charts.
    metrics : Dict
        Stores metric calculation function and weight for each chart.

    Returns
    -------
    combined_score: float
        Combined weighted average score.
    scores: list
        List of scores for each chart.
    """
    scores = {}
    for chart in charts:
        if true[chart].ndim == 1 and pred[chart].ndim == 1:
            scores[chart] = torch.round(metrics[chart]['func'](
                true=true[chart], pred=pred[chart], num_classes=num_classes[chart]) * 100, decimals=3)

        else:
            print(f"true and pred must be 1D numpy array, got {true['SIC'].ndim} \
                and {pred['SIC'].ndim} dimensions with shape {true['SIC'].shape} and {pred.shape}, respectively")

    combined_score = compute_combined_score(scores=scores, charts=charts, metrics=metrics)

    return combined_score, scores


def r2_metric(true, pred, num_classes=None):
    """
    Calculate the r2 metric.

    Parameters
    ----------
    true :
        ndarray, 1d contains all true pixels. Must by numpy array.
    pred :
        ndarray, 1d contains all predicted pixels. Must by numpy array.
    num_classes :
        Num of classes in the dataset, this value is not used in this function but used in f1_metric function
        which requires num_classes argument. The reason it was included here was to keep the same structure.  


    Returns
    -------
    r2 : float
        The calculated r2 score.

    """
    r2 = r2_score(preds=pred, target=true)

    return r2


def f1_metric(true, pred, num_classes):
    """
    Calculate the weighted f1 metric.

    Parameters
    ----------
    true :
        ndarray, 1d contains all true pixels.
    pred :
        ndarray, 1d contains all predicted pixels.

    Returns
    -------
    f1 : float
        The calculated f1 score.

    """
    f1 = f1_score(target=true, preds=pred, average='weighted', task='multiclass', num_classes=num_classes)

    return f1


def water_edge_metric(outputs, options):
    # 需要至少3个任务才能计算跨任务水体一致性；单任务模式返回0
    if len(options['charts']) < 3:
        return torch.tensor(0.0)

    # Convert ouput into water and not water
    for chart in options['charts']:
        outputs[chart] = torch.where(outputs[chart] > 0.0, 1.0, 0.0)

    water_edge_accuracy = 1 - torch.mean(torch.abs(outputs[options['charts'][0]]-outputs[options['charts'][1]])
                                         + torch.abs(outputs[options['charts'][1]]-outputs[options['charts'][2]])
                                         + torch.abs(outputs[options['charts'][2]]-outputs[options['charts'][0]]))
    return water_edge_accuracy


def water_edge_plot_overlay(output, mask, options):
    # Convert ouput into water and not water
    charts = options['charts']
    water_chart = {}
    for chart in charts:
        water_chart[chart] = np.where(output[chart] > 0.0, 0.75, 0.0)
        water_chart[chart][mask] = np.nan
        water_chart[chart] = water_chart[chart][..., np.newaxis]

    img = np.concatenate((water_chart[charts[0]], water_chart[charts[1]], water_chart[charts[2]]), axis=2,)

    return img


def compute_combined_score(scores, charts, metrics):
    """
    Calculate the combined weighted score.

    Parameters
    ----------
    scores : List
        Score for each chart.
    charts : List
        List of charts.
    metrics : Dict
        Stores metric calculation function and weight for each chart.

    Returns
    -------
    : float
        The combined weighted score.

    """
    combined_metric = 0
    sum_weight = 0
    for chart in charts:
        combined_metric += scores[chart] * metrics[chart]['weight']
        sum_weight += metrics[chart]['weight']

    return torch.round(combined_metric / sum_weight, decimals=3)


# -- functions to save models -- #
def save_best_model(cfg, train_options: dict, net, optimizer, scheduler, epoch: int):
    '''
    Saves the input model in the inside the directory "/work_dirs/"experiment_name"/
    The models with be save as best_model.pth.
    The following are stored inside best_model.pth
        model_state_dict
        optimizer_state_dict
        epoch
        train_options


    Parameters
    ----------
    cfg : mmcv.Config
        The config file object of mmcv
    train_options : Dict
        The dictory which stores the train_options from quickstart
    net :
        The pytorch model
    optimizer :
        The optimizer that the model uses.
    epoch: int
        The epoch number

    '''
    print('saving model....')
    config_file_name = os.path.basename(cfg.work_dir)
    # print(config_file_name)
    torch.save(obj={'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'train_options': train_options
                    },
               f=os.path.join(cfg.work_dir, f'best_model_{config_file_name}.pth'))
    print(f"model saved successfully at {os.path.join(cfg.work_dir, f'best_model_{config_file_name}.pth')}")

    return os.path.join(cfg.work_dir, f'best_model_{config_file_name}.pth')


def load_model(net, checkpoint_path, optimizer=None, scheduler=None):
    """
    Loads a PyTorch model from a checkpoint file and returns the model, optimizer, and scheduler.
    :param model: PyTorch model to load
    :param checkpoint_path: Path to the checkpoint file
    :param optimizer: PyTorch optimizer to load (optional)
    :param scheduler: PyTorch scheduler to load (optional)
    :return: If optimizer and scheduler are provided, return the model, optimizer, and scheduler.
    """

    # 修改此处，添加 weights_only=False
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
    except TypeError:
        # 如果是旧版本 PyTorch 不支持 weights_only 参数，则回退到默认行为
        checkpoint = torch.load(checkpoint_path)

    net.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']

    return epoch

def rand_bbox(size, lam):
    '''
    Given the 4D dimensions of a batch (size), and the ratio 
    of the spatial dimension (lam) to be cut, returns a bounding box coordinates
    used for cutmix

    Parameters
    ----------
    size : 4D shape of the batch (N, C, H, W)
    lam : Ratio (portion) of the input to be cutmix'd

    Returns 
    ----------
    Bounding box (x1, y1, x2, y2)
    '''
    H = size[2]
    W = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)

    # uniform
    cx = np.random.randint(H)
    cy = np.random.randint(W)

    bbx1 = np.clip(cx - cut_h // 2, 0, H)
    bby1 = np.clip(cy - cut_w // 2, 0, W)
    bbx2 = np.clip(cx + cut_h // 2, 0, H)
    bby2 = np.clip(cy + cut_w // 2, 0, W)

    return bbx1, bby1, bbx2, bby2


def slide_inference(img, net, options, mode):
    """
    Inference by sliding-window with overlap.


    Parameters
    ----------
    img : 4D shape of the batch (N, C', H, W)
    net : PyTorch model of nn.Module 
    options: configuration dictionary
    mode: either 'val' or 'test'

    Returns 
    ----------
    pred: Dictionary with SIC, SOD, and FLOE predictions of the batch  (N, C", H, W)
    """
    if mode == 'val':
        h_stride, w_stride = options['swin_hp']['val_stride']
    elif mode == 'test':
        h_stride, w_stride = options['swin_hp']['test_stride']
    else:
        raise 'Unrecognized mode'

    h_crop = options['patch_size']
    w_crop = options['patch_size']

    batch_size, _, h_img, w_img = img.size()
    device = img.device
    charts = options['charts']
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

    # 累积张量放 CPU，避免大场景下 GPU OOM；每个 crop 的 forward 仍在 GPU 上执行
    preds = {chart: torch.zeros((batch_size, options['n_classes'][chart], h_img, w_img),
                                dtype=img.dtype) for chart in charts}
    count_mat = torch.zeros((batch_size, 1, h_img, w_img), dtype=img.dtype)

    total_patches = h_grids * w_grids
    patch_bar = tqdm(total=total_patches, desc='slide patches', leave=False,
                     disable=total_patches < 20)
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            patch_bar.update(1)
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = img[:, :, y1:y2, x1:x2]
            crop_img_size = crop_img.size()
            crop_height_pad = max(options['patch_size'] - crop_img_size[2], 0)
            crop_width_pad  = max(options['patch_size'] - crop_img_size[3], 0)

            if crop_height_pad > 0 or crop_width_pad > 0:
                crop_img = torch.nn.functional.pad(
                    crop_img, (0, crop_width_pad, 0, crop_height_pad), mode='constant', value=0)

            crop_seg_logit = net(crop_img)

            for chart in charts:
                logit = crop_seg_logit[chart]
                if crop_height_pad > 0:
                    logit = logit[:, :, :-crop_height_pad, :]
                if crop_width_pad > 0:
                    logit = logit[:, :, :, :-crop_width_pad]
                # pad 到全场景尺寸后移至 CPU 累积，避免 GPU 上创建全场景临时张量
                preds[chart][:, :, y1:y2, x1:x2] += logit.cpu()

            count_mat[:, :, y1:y2, x1:x2] += 1

    patch_bar.close()
    assert (count_mat == 0).sum() == 0

    # 归一化后送回 GPU
    return {chart: (preds[chart] / count_mat).to(device) for chart in charts}


class Slide_patches_index(data.Dataset):
    def __init__(self, h_img, w_img, h_crop, w_crop, h_stride, w_stride):
        super(Slide_patches_index, self).__init__()

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        self.patches_list = []

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                self.patches_list.append((y1, y2, x1, x2))

    def __getitem__(self, index):
        return self.patches_list[index]

    def __len__(self):
        return len(self.patches_list)


class Take_crops(data.Dataset):
    def __init__(self, img, patches):
        super(Take_crops, self).__init__()

        self.img = img
        self.patches = patches

    def __getitem__(self, index):
        y1, y2, x1, x2 = self.patches[index]

        return self.img[:, y1:y2, x1:x2]

    def __len__(self):
        return len(self.patches)


def batched_slide_inference(img, net, options, mode):
    """
    Inference by sliding-window with overlap.

    Parameters
    ----------
    img : 4D shape of the batch (N, C', H, W)
    net : PyTorch model of nn.Module 
    y_type: str, One of 'SIC', 'SOD', or 'FLOE'
    options: configuration dictionary

    Returns 
    ----------
    pred: Dictionary with SIC, SOD, and FLOE predictions of the batch  (N, C", H, W)
    """
    if mode == 'val':
        h_stride, w_stride = options['swin_hp']['val_stride']
    elif mode == 'test':
        h_stride, w_stride = options['swin_hp']['test_stride']
    else:
        raise 'Unrecognized mode'

    h_crop = options['patch_size']
    w_crop = options['patch_size']

    # ------------ Add Padding to the image to match with the patch size / stride
    _, _, h_img, w_img = img.size()
    height_pad = h_crop - h_img if h_img - h_crop < 0 else \
        (h_stride - (h_img - h_crop) % h_stride) % h_stride
    width_pad = w_crop - w_img if w_img - w_crop < 0 else \
        (w_stride - (w_img - w_crop) % w_stride) % w_stride
    if height_pad > 0 or width_pad > 0:
        img = torch.nn.functional.pad(
            img, (0, width_pad, 0, height_pad), mode='constant', value=0)

    # ------------ create dataloader and index track
    _, _, h_img, w_img = img.size()
    indexes = Slide_patches_index(h_img, w_img, h_crop, w_crop, h_stride, w_stride)
    samples = Take_crops(img.detach().cpu().numpy()[0], indexes.patches_list)
    samples_dataloader = data.DataLoader(dataset=samples, batch_size=options['batch_size']*4,
                                         shuffle=False, num_workers=options['num_workers_val'])

    n_batches = len(samples_dataloader)
    data_iterator = iter(samples_dataloader)
    idx_iterator = iter(indexes)

    SIC_channels = options['n_classes']['SIC']
    SOD_channels = options['n_classes']['SOD']
    FLOE_channels = options['n_classes']['FLOE']
    preds_SIC = img.new_zeros((SIC_channels, h_img, w_img))
    preds_SOD = img.new_zeros((SOD_channels, h_img, w_img))
    preds_FLOE = img.new_zeros((FLOE_channels, h_img, w_img))
    count_mat = img.new_zeros((h_img, w_img))

    for i in range(n_batches):

        # ------------ Take data
        crop_imgs = next(data_iterator)
        crop_imgs = crop_imgs.to(img.device)

        # ------------ Forward
        crop_seg_logit = net(crop_imgs)

        # ------------ LOCATE PREDICTED LOGITS ON THE WHOLE SCENE
        for j in range(crop_imgs.shape[0]):
            y1, y2, x1, x2 = next(idx_iterator)

            preds_SIC[:, y1:y2, x1:x2] += crop_seg_logit['SIC'][j, :, 0:(y2-y1), 0:(x2-x1)]
            preds_SOD[:, y1:y2, x1:x2] += crop_seg_logit['SOD'][j, :, 0:(y2-y1), 0:(x2-x1)]
            preds_FLOE[:, y1:y2, x1:x2] += crop_seg_logit['FLOE'][j, :, 0:(y2-y1), 0:(x2-x1)]

            count_mat[y1:y2, x1:x2] += 1

    assert (count_mat == 0).sum() == 0

    preds_SIC = preds_SIC / count_mat
    preds_SOD = preds_SOD / count_mat
    preds_FLOE = preds_FLOE / count_mat

    # ------------ Remove pad (guard against 0-pad: slicing with :-0 yields empty dimension)
    if height_pad > 0:
        preds_SIC = preds_SIC[:, :-height_pad, :]
        preds_SOD = preds_SOD[:, :-height_pad, :]
        preds_FLOE = preds_FLOE[:, :-height_pad, :]
    if width_pad > 0:
        preds_SIC = preds_SIC[:, :, :-width_pad]
        preds_SOD = preds_SOD[:, :, :-width_pad]
        preds_FLOE = preds_FLOE[:, :, :-width_pad]

    preds_SIC = preds_SIC.unsqueeze(0)
    preds_SOD = preds_SOD.unsqueeze(0)
    preds_FLOE = preds_FLOE.unsqueeze(0)

    return {'SIC': preds_SIC,
            'SOD': preds_SOD,
            'FLOE': preds_FLOE}


def fast_tiled_val_inference(img, net, options):
    """
    Fast validation inference using non-overlapping tiles with batched GPU forward passes.

    Splits the scene into non-overlapping patch_size×patch_size tiles, batches them
    together, and runs a single net() call per batch. Dramatically faster than
    sliding-window inference for full-resolution (scale=1) scenes:
      e.g. 2560×2560, patch_size=256, val_batch_size=256 → 1 forward pass/scene
           vs. 484 forward passes with stride-128 sliding window.

    Caller is responsible for wrapping in torch.no_grad() / autocast().

    Parameters
    ----------
    img : Tensor (1, C, H, W)  on the target device
    net : nn.Module
    options : dict
        Uses 'patch_size', 'charts', 'n_classes', 'batch_size',
        and optionally 'val_batch_size' (defaults to batch_size * 4).

    Returns
    -------
    dict[str, Tensor (1, n_classes[chart], H, W)]  on the same device as img.
    """
    patch_size = options['patch_size']
    charts = options['charts']
    val_batch_size = options.get('val_batch_size', options['batch_size'] * 4)
    device = img.device

    _, C, h_img, w_img = img.size()

    # Pad H and W to exact multiples of patch_size
    h_pad = (-h_img) % patch_size
    w_pad = (-w_img) % patch_size
    if h_pad > 0 or w_pad > 0:
        img = torch.nn.functional.pad(img, (0, w_pad, 0, h_pad), mode='constant', value=0)
    _, _, h_padded, w_padded = img.size()

    n_h = h_padded // patch_size
    n_w = w_padded // patch_size
    n_tiles = n_h * n_w

    # Rearrange padded scene into non-overlapping tiles (all on CPU to save GPU memory)
    # (C, h_padded, w_padded) → (n_tiles, C, patch_size, patch_size)
    img_cpu = img[0].cpu()
    tiles = (img_cpu
             .reshape(C, n_h, patch_size, n_w, patch_size)
             .permute(1, 3, 0, 2, 4)
             .reshape(n_tiles, C, patch_size, patch_size)
             .contiguous())

    # Prediction buffers on CPU; only each batch touches the GPU.
    # CORAL head outputs K-1 channels for SOD; all other charts keep K channels.
    def _out_ch(chart):
        K = options['n_classes'][chart]
        return K - 1 if (chart == 'SOD' and options.get('sod_head') == 'coral') else K

    preds = {chart: torch.zeros(_out_ch(chart), h_padded, w_padded)
             for chart in charts}

    for start in range(0, n_tiles, val_batch_size):
        end = min(start + val_batch_size, n_tiles)
        batch = tiles[start:end].to(device)
        batch_out = net(batch)

        for local_j, tile_idx in enumerate(range(start, end)):
            h_idx = tile_idx // n_w
            w_idx = tile_idx % n_w
            y1 = h_idx * patch_size
            x1 = w_idx * patch_size
            for chart in charts:
                preds[chart][:, y1:y1 + patch_size, x1:x1 + patch_size] = \
                    batch_out[chart][local_j].detach().cpu()

    # Trim padding, restore batch dim, return on original device
    return {chart: preds[chart][:, :h_img, :w_img].unsqueeze(0).to(device)
            for chart in charts}


def class_decider(output, train_options, chart):

    # normal
    if (train_options['binary_water_classifier'] == False):
        if output.size(3) == 1:
            output = torch.round(output.squeeze())
            output = torch.clamp(output, min=0, max=train_options
                                 ['n_classes'][chart])
            return output

        # CORAL head outputs K-1 sigmoid CDF values; convert to K class probs first.
        if train_options.get('sod_head') == 'coral' and chart == 'SOD':
            probs = coral_cdf_to_probs(output.float())  # [B, K, H, W]
        else:
            probs = None

        decision_mode = train_options.get('inference_decision', 'argmax')
        if decision_mode == 'ordinal_risk_optimal':
            if probs is None:
                probs = torch.softmax(output.float(), dim=1)
            return ordinal_argmin_decision(probs, NAVIGATION_COST_MATRIX).squeeze(0)
        elif decision_mode == 'cvar_risk_optimal':
            if probs is None:
                probs = torch.softmax(output.float(), dim=1)
            q = train_options.get('cvar_q', 1.0)
            return cvar_optimal_decision(probs, NAVIGATION_COST_MATRIX, q=q).squeeze(0)
        else:
            if probs is not None:
                return torch.argmax(probs, dim=1).squeeze(0)
            return torch.argmax(output, dim=1).squeeze(0)

    # if regression head    return output
    # class water
    else:
        probability = torch.nn.Softmax(dim=1)(output)
        water = probability[:, 0, :, :]
        not_water = torch.sum(probability, dim=1) - water
        class_output = water <= not_water
        without_water = probability[:, 1:, :, :]
        class_output_without_water = torch.argmax(without_water, dim=1) + 1
        class_output = class_output_without_water * class_output

        return class_output.squeeze(0)


def compute_classwise_f1score(true, pred, charts, num_classes):
    """ This function computes the classwise evaluation score for each task and stores them in a dic

    Args:
        true (dictionary): The true tensor as value and chart tensor as key
        pred (dictionary): The pred tensor as value and chart tensor as key
        charts (list): list of charts
        num_classes (dictionary): key = chart , value = num_class

    Returns:
        dictionary: returns score_dictionary
    """
    score = {}
    for chart in charts:
        score[chart] = f1_score(target=true[chart], preds=pred[chart], average='none',
                                task='multiclass', num_classes=num_classes[chart])
    return score


def compute_overall_accuracy(true, pred, charts):
    """ Computes Overall Accuracy (OA) for each task.

    Args:
        true (dictionary): The true tensor as value and chart tensor as key
        pred (dictionary): The pred tensor as value and chart tensor as key
        charts (list): list of charts

    Returns:
        dictionary: returns oa_score_dictionary
    """
    scores = {}
    for chart in charts:
        if true[chart].numel() == 0:
            scores[chart] = 0.0
        else:
            # 计算预测正确的像素数占总像素数的比例
            correct = (true[chart] == pred[chart]).sum()
            total = true[chart].numel()
            scores[chart] = (correct.float() / total).item()
    return scores


def compute_mIoU(true, pred, charts, num_classes):
    """ Computes Mean Intersection over Union (mIoU) for each task.

    Args:
        true (dictionary): The true tensor as value and chart tensor as key
        pred (dictionary): The pred tensor as value and chart tensor as key
        charts (list): list of charts
        num_classes (dictionary): key = chart , value = num_class

    Returns:
        dictionary: returns mIoU_score_dictionary
    """
    scores = {}
    for chart in charts:
        # jaccard_index 即为 IoU。task='multiclass' 适用于多分类，默认 average='macro' 即计算 mIoU
        scores[chart] = jaccard_index(target=true[chart], preds=pred[chart], task='multiclass', num_classes=num_classes[chart])
    return scores


def compute_ordinal_cost_metric(true, pred, charts, num_classes, cost_per_distance=None):
    """Ordinal Cost Metric for ordered classification (e.g. SOD thickness stages).

    Per-pixel cost is looked up from cost_per_distance[|pred - true|].
    Default is linear: cost = |pred - true|, which is consistent with EMDLoss.

    Parameters
    ----------
    cost_per_distance : list of numbers, length >= n_classes.
        cost_per_distance[d] is the penalty for a prediction that is d classes away.
        Must start with 0 (correct prediction costs nothing).
        Example (linear):  [0, 1, 2, 3, 4]
        Example (custom):  [0, 1, 2, 3, 3]  (cap at 3 for very distant errors)

    Returns
    -------
    raw_costs : dict[chart -> Tensor scalar]  mean cost per pixel (lower = better)
    norm_scores : dict[chart -> Tensor scalar]
        1 - mean_cost / max_cost, normalised to [0, 1] (higher = better).
    """
    raw_costs = {}
    norm_scores = {}
    for chart in charts:
        n_classes = num_classes[chart]
        true_c = true[chart].long()
        pred_c = pred[chart].long()

        valid = (true_c >= 0) & (true_c < n_classes) & (pred_c >= 0) & (pred_c < n_classes)
        true_valid = true_c[valid]
        pred_valid = pred_c[valid]

        if true_valid.numel() == 0:
            raw_costs[chart] = torch.tensor(0.0)
            norm_scores[chart] = torch.tensor(1.0)
            continue

        # Build cost lookup table: index = distance, value = penalty
        if cost_per_distance is None:
            # Linear default: cost = |pred - true|, aligned with EMDLoss
            cost_table = torch.arange(n_classes, dtype=torch.float32, device=true_c.device)
        else:
            cost_table = torch.tensor(cost_per_distance, dtype=torch.float32, device=true_c.device)

        diff = torch.abs(true_valid - pred_valid)          # [N], integer distances
        cost = cost_table[diff]                            # lookup per-pixel cost

        max_cost = cost_table.max().clamp(min=1e-6)
        mean_cost = cost.mean()
        raw_costs[chart] = mean_cost
        norm_scores[chart] = 1.0 - mean_cost / max_cost

    return raw_costs, norm_scores


def compute_navigation_risk_metrics(true, pred, charts, num_classes,
                                    cost_matrix=None):
    """Compute NRS, OVR, and UR using the asymmetric navigation cost matrix.

    Returns
    -------
    nrs : dict[chart -> Tensor]  Navigation Risk Score  (higher is better)
    ovr : dict[chart -> Tensor]  Ordinal Violation Rate (lower is better)
    ur  : dict[chart -> Tensor]  Underestimation Rate   (lower is better)
    """
    if cost_matrix is None:
        cost_matrix = NAVIGATION_COST_MATRIX

    nrs, ovr, ur = {}, {}, {}
    for chart in charts:
        n_cls  = num_classes[chart]
        true_c = true[chart].long()
        pred_c = pred[chart].long()
        valid  = (true_c >= 0) & (true_c < n_cls) & (pred_c >= 0) & (pred_c < n_cls)
        t, p   = true_c[valid], pred_c[valid]

        if t.numel() == 0:
            nrs[chart] = torch.tensor(1.0)
            ovr[chart] = torch.tensor(0.0)
            ur[chart]  = torch.tensor(0.0)
            continue

        cm         = cost_matrix.to(t.device)
        costs      = cm[t, p]
        nrs[chart] = 1.0 - costs.mean() / cm.max().clamp(min=1e-6)
        ovr[chart] = (torch.abs(t - p) >= 2).float().mean()
        ur[chart]  = (p < t).float().mean()

    return nrs, ovr, ur


def ordinal_argmin_decision(probs: torch.Tensor,
                            cost_matrix: torch.Tensor) -> torch.Tensor:
    """Optimal class decision minimising expected cost under cost_matrix.

    probs       : [B, K, H, W]  softmax probabilities (float32)
    cost_matrix : [K, K]        C[true, pred]

    Returns [B, H, W] long tensor of optimal class indices.
    """
    cm = cost_matrix.to(probs.device).float()              # [K, K]
    expected_cost = torch.einsum('bkhw,kj->bjhw', probs.float(), cm)  # [B, K, H, W]
    return expected_cost.argmin(dim=1)                     # [B, H, W]


def cvar_optimal_decision(probs: torch.Tensor,
                          cost_matrix: torch.Tensor,
                          q: float = 1.0) -> torch.Tensor:
    """Risk-averse class decision minimising CVaR_q of the navigation cost.

    For each candidate decision j, the cost is a random variable taking value
    C[k, j] with probability p_k over the true class k.  CVaR_q is the mean of
    the worst q-probability tail of that cost distribution.  Minimising it makes
    the decoder hedge against low-probability, high-cost true classes (e.g. a
    small chance of multi-year ice when the cost of calling it open water is
    huge), which expected-cost decoding can wash out.

    Risk-aversion is controlled by the tail fraction q in (0, 1]:
        q = 1.0  recovers expected-cost decoding (== ordinal_argmin_decision);
        q -> 0   approaches the minimax (worst-case true class) decision.

    The cost-ordering of true classes for a fixed decision j depends only on
    column j of the cost matrix, not on the pixel, so the per-class tail weights
    are obtained without any per-pixel sort.

    probs       : [B, K, H, W]  softmax probabilities (float32)
    cost_matrix : [K, K]        C[true, pred]
    q           : float in (0, 1]   tail fraction (risk-aversion level)

    Returns [B, H, W] long tensor of optimal class indices.
    """
    if not 0.0 < q <= 1.0:
        raise ValueError(f"cvar q must be in (0, 1], got {q}")
    if q == 1.0:                                           # risk-neutral special case
        return ordinal_argmin_decision(probs, cost_matrix)

    cm = cost_matrix.to(probs.device).float()              # [K, K]  C[true, pred]
    P = probs.float()                                      # [B, K, H, W]  prob over true class
    K = cm.shape[1]

    # Streaming accumulation keeps only [B, H, W] tensors (never [B, H, W, K]),
    # which matters on full-resolution scenes where the K axis would blow up VRAM.
    best_cost, best_j = None, None
    for j in range(K):
        costs_j = cm[:, j]                                 # [K]  cost over true classes
        order = torch.argsort(costs_j, descending=True)    # worst-cost-first permutation
        cum = torch.zeros_like(P[:, 0])                    # [B, H, W]  cumulative prob mass
        acc = torch.zeros_like(P[:, 0])                    # [B, H, W]  accumulated tail cost
        for k in order.tolist():
            new_cum = cum + P[:, k]
            # portion of class k's mass that falls within the worst q tail
            take = new_cum.clamp(max=q) - cum.clamp(max=q)
            acc = acc + take * costs_j[k]
            cum = new_cum
        cvar_j = acc / q                                   # [B, H, W]  CVaR_q for decision j
        if best_cost is None:
            best_cost = cvar_j
            best_j = torch.zeros_like(cvar_j, dtype=torch.long)
        else:                                              # strict < keeps lowest j on ties
            better = cvar_j < best_cost
            best_j = torch.where(better, torch.full_like(best_j, j), best_j)
            best_cost = torch.where(better, cvar_j, best_cost)

    return best_j                                          # [B, H, W]


def coral_cdf_to_probs(cdf_gt: torch.Tensor) -> torch.Tensor:
    """Convert CORALHead output P(y > k) to class probabilities P(y = k).

    cdf_gt  : (B, K-1, H, W)  — P(y > k) from CORALHead sigmoid output
    returns : (B, K,   H, W)  — P(y = k), clamped ≥ 0 (monotonicity not guaranteed)
    """
    p0    = 1.0 - cdf_gt[:, :1, :, :]                    # P(y=0) = 1 - P(y>0)
    p_mid = cdf_gt[:, :-1, :, :] - cdf_gt[:, 1:, :, :]  # P(y=k) = P(y>k-1) - P(y>k)
    p_K   = cdf_gt[:, -1:, :, :]                          # P(y=K-1) = P(y>K-2)
    return torch.cat([p0, p_mid, p_K], dim=1).clamp(min=0.0)


def compute_classwise_IoU(true, pred, charts, num_classes):
    """Computes per-class IoU for each task.

    Args:
        true (dictionary): The true tensor as value and chart tensor as key
        pred (dictionary): The pred tensor as value and chart tensor as key
        charts (list): list of charts
        num_classes (dictionary): key = chart , value = num_class

    Returns:
        dictionary: per-class IoU tensor for each chart
    """
    scores = {}
    for chart in charts:
        scores[chart] = jaccard_index(
            target=true[chart], preds=pred[chart],
            task='multiclass', num_classes=num_classes[chart],
            average='none'
        )
    return scores


def compute_classwise_precision_recall(true, pred, charts, num_classes):
    """Computes per-class precision and recall for each chart.

    Invalid labels (outside [0, num_classes-1]) are ignored.
    """
    precision_scores = {}
    recall_scores = {}

    for chart in charts:
        n_classes = num_classes[chart]
        true_c = true[chart].long()
        pred_c = pred[chart].long()

        valid = (true_c >= 0) & (true_c < n_classes) & (pred_c >= 0) & (pred_c < n_classes)
        true_valid = true_c[valid]
        pred_valid = pred_c[valid]

        if true_valid.numel() == 0:
            precision_scores[chart] = torch.zeros(n_classes, dtype=torch.float32, device=true_c.device)
            recall_scores[chart] = torch.zeros(n_classes, dtype=torch.float32, device=true_c.device)
            continue

        cm = multiclass_confusion_matrix(
            preds=pred_valid,
            target=true_valid,
            num_classes=n_classes,
        ).float()

        tp = torch.diag(cm)
        pred_sum = cm.sum(dim=0)
        true_sum = cm.sum(dim=1)

        precision_scores[chart] = torch.where(pred_sum > 0, tp / pred_sum, torch.zeros_like(tp))
        recall_scores[chart] = torch.where(true_sum > 0, tp / true_sum, torch.zeros_like(tp))

    return precision_scores, recall_scores


def compute_normalized_confusion_matrix(true, pred, charts, num_classes):
    """Computes row-normalized confusion matrix for each chart.

    Normalization is done by true-class row sums.
    """
    norm_cms = {}
    for chart in charts:
        n_classes = num_classes[chart]
        true_c = true[chart].long()
        pred_c = pred[chart].long()

        valid = (true_c >= 0) & (true_c < n_classes) & (pred_c >= 0) & (pred_c < n_classes)
        true_valid = true_c[valid]
        pred_valid = pred_c[valid]

        if true_valid.numel() == 0:
            norm_cms[chart] = torch.zeros((n_classes, n_classes), dtype=torch.float32, device=true_c.device)
            continue

        cm = multiclass_confusion_matrix(
            preds=pred_valid,
            target=true_valid,
            num_classes=n_classes,
        ).float()

        row_sum = cm.sum(dim=1, keepdim=True)
        norm_cms[chart] = torch.where(row_sum > 0, cm / row_sum, torch.zeros_like(cm))

    return norm_cms


def plot_cost_weighted_confusion_matrix(norm_cms, charts, num_classes,
                                        cost_matrix=None, save_dir=None, epoch=None):
    """Generate side-by-side standard vs cost-weighted confusion matrix plots.

    Parameters
    ----------
    norm_cms  : dict[chart -> Tensor [K, K]]  row-normalised confusion matrix
    save_dir  : if given, saves PNG files there
    epoch     : epoch number for filename / title

    Returns
    -------
    figs : dict[chart -> matplotlib.figure.Figure]
    """
    if cost_matrix is None:
        cost_matrix = NAVIGATION_COST_MATRIX

    _SOD_LABELS = ['Water', 'New/Young', 'Thin FYI', 'Thick FYI', 'MYI']
    _CHART_LABELS = {'SOD': _SOD_LABELS}

    figs = {}
    for chart in charts:
        if chart not in norm_cms:
            continue
        cm_norm = norm_cms[chart].cpu().numpy()
        K = cm_norm.shape[0]
        C = cost_matrix[:K, :K].cpu().numpy()
        cm_cost = cm_norm * C
        labels = _CHART_LABELS.get(chart, [str(i) for i in range(K)])

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{chart} — Standard vs Cost-Weighted Confusion Matrix  (epoch {epoch})',
                     fontsize=12)

        for ax, data, title, cmap in [
            (axes[0], cm_norm, 'Standard  (row-normalised)', 'Blues'),
            (axes[1], cm_cost, 'Cost-Weighted  (CM ⊙ C)',   'Reds'),
        ]:
            im = ax.imshow(data, interpolation='nearest', cmap=cmap,
                           vmin=0, vmax=data.max() if data.max() > 0 else 1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel('Predicted class')
            ax.set_ylabel('True class')
            ax.set_xticks(range(K))
            ax.set_yticks(range(K))
            ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
            ax.set_yticklabels(labels, fontsize=8)
            thresh = data.max() / 2.0
            for i in range(K):
                for j in range(K):
                    fmt = '.2f' if 'norm' in title.lower() else '.3f'
                    ax.text(j, i, format(data[i, j], fmt),
                            ha='center', va='center', fontsize=7,
                            color='white' if data[i, j] > thresh else 'black')

        plt.tight_layout()

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fname = os.path.join(save_dir, f'{chart}_cost_cm_ep{epoch:03d}.png')
            fig.savefig(fname, dpi=120, bbox_inches='tight')

        figs[chart] = fig

    return figs


def compute_class_proportions(true, pred, charts, num_classes):
    """Computes per-class true/pred pixel proportions for each chart."""
    true_props = {}
    pred_props = {}

    for chart in charts:
        n_classes = num_classes[chart]
        true_c = true[chart].long()
        pred_c = pred[chart].long()

        valid = (true_c >= 0) & (true_c < n_classes) & (pred_c >= 0) & (pred_c < n_classes)
        true_valid = true_c[valid]
        pred_valid = pred_c[valid]

        if true_valid.numel() == 0:
            true_props[chart] = torch.zeros(n_classes, dtype=torch.float32, device=true_c.device)
            pred_props[chart] = torch.zeros(n_classes, dtype=torch.float32, device=true_c.device)
            continue

        true_hist = torch.bincount(true_valid, minlength=n_classes).float()
        pred_hist = torch.bincount(pred_valid, minlength=n_classes).float()
        denom = true_hist.sum().clamp_min(1.0)

        true_props[chart] = true_hist / denom
        pred_props[chart] = pred_hist / denom

    return true_props, pred_props


def create_train_validation_and_test_scene_list(train_options):
    '''
    Creates the train, validation, and test scene lists from JSON datalist files.
    For MyDS, the filenames in the JSON files are the actual .nc filenames
    (no string manipulation needed).
    '''

    # Train ------------
    with open(train_options['path_to_env'] + train_options['train_list_path']) as file:
        train_options['train_list'] = json.loads(file.read())

    # Validation ---------
    if train_options['cross_val_run']:
        train_options['validate_list'] = list(np.random.choice(np.array(
            train_options['train_list']), size=train_options['p-out'], replace=False))
    else:
        with open(train_options['path_to_env'] + train_options['val_path']) as file:
            train_options['validate_list'] = json.loads(file.read())

    # Remove the validation scenes from the train list.
    train_options['train_list'] = [scene for scene in train_options['train_list']
                                   if scene not in train_options['validate_list']]

    # Test ----------
    with open(train_options['path_to_env'] + train_options['test_path']) as file:
        train_options['test_list'] = json.loads(file.read())
    # For MyDS, labels are embedded in the same file - reference list equals test list.
    train_options['test_list_reference'] = train_options['test_list']

    print('Options initialised')


def get_scheduler(train_options, optimizer):
    if train_options['scheduler']['type'] == 'CosineAnnealingLR':
        T_max_epochs = train_options['scheduler'].get('T_max_epochs', train_options['epochs'])
        T_max = T_max_epochs * train_options['epoch_len']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max,
                                                               eta_min=train_options['scheduler']['lr_min'])
    elif train_options['scheduler']['type'] == 'CosineAnnealingWarmRestartsLR':
        # T_max = train_options['epochs']*train_options['epoch_len']
        T_0 = train_options['scheduler']['EpochsPerRestart']*train_options['epoch_len']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0,
                                                                         T_mult=train_options['scheduler']['RestartMult'],
                                                                         eta_min=train_options['scheduler']['lr_min'],
                                                                         last_epoch=-1
                                                                        #  verbose=False
                                                                         )
    elif train_options['scheduler']['type'] == 'ReduceLROnPlateau':
        # 每 epoch 验证后调用 scheduler.step(val_score)，无需 per-batch 步进
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',                                               # 监控的指标越大越好（combined_score）
            factor=train_options['scheduler'].get('factor', 0.5),    # 触发时 lr *= factor
            patience=train_options['scheduler'].get('patience', 5),  # 连续多少个验证epoch无改善后触发
            min_lr=train_options['scheduler'].get('min_lr', 1e-5),   # lr 下限
            threshold=train_options['scheduler'].get('threshold', 1e-4),  # 判定"改善"的最小变化量
        )
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=5, last_epoch=- 1,
                                                        # verbose=False
                                                        )
    return scheduler


def get_optimizer(train_options, net):
    if train_options['optimizer']['type'] == 'Adam':
        optimizer = torch.optim.Adam(list(net.parameters()), lr=train_options['optimizer']['lr'],
                                     betas=(train_options['optimizer']['b1'], train_options['optimizer']['b2']),
                                     weight_decay=train_options['optimizer']['weight_decay'])

    elif train_options['optimizer']['type'] == 'AdamW':
        optimizer = torch.optim.AdamW(list(net.parameters()), lr=train_options['optimizer']['lr'],
                                      betas=(train_options['optimizer']['b1'], train_options['optimizer']['b2']),
                                      weight_decay=train_options['optimizer']['weight_decay'])
    else:
        optimizer = torch.optim.SGD(list(net.parameters()), lr=train_options['optimizer']['lr'],
                                    momentum=train_options['optimizer']['momentum'],
                                    dampening=train_options['optimizer']['dampening'],
                                    weight_decay=train_options['optimizer']['weight_decay'],
                                    nesterov=train_options['optimizer']['nesterov'])
    return optimizer


def get_loss(loss, chart=None, **kwargs):
    # TODO Fix Dice loss, Jacard loss,  MCC loss, SoftBCEWithLogitsLoss,
    """_summary_

    Args:
        loss (str): the name of the loss
    Returns:
        loss: The corresponding
    """
    if loss == 'DiceLoss':
        kwargs.pop('type')
        loss = smp.losses.DiceLoss(**kwargs)
    elif loss == 'FocalLoss':
        from losses import FocalLoss as _FocalLoss
        kwargs.pop('type')
        loss = _FocalLoss(**kwargs)
    elif loss == 'JaccardLoss':
        raise NotImplementedError
        kwargs.pop('type')
        loss = smp.losses.JaccardLoss(**kwargs)
    elif loss == 'LovaszLoss':
        kwargs.pop('type')
        loss = smp.losses.LovaszLoss(**kwargs)
    elif loss == 'MCCLoss':
        kwargs.pop('type')
        loss = smp.losses.MCCLoss(**kwargs)
    elif loss == 'SoftBCEWithLogitsLoss':
        raise NotImplementedError
        kwargs.pop('type')
        loss = smp.losses.SoftBCEWithLogitsLoss(**kwargs)
    elif loss == 'SoftCrossEntropyLoss':
        raise NotImplementedError
        kwargs.pop('type')
        loss = smp.losses.SoftCrossEntropyLoss(**kwargs)
    elif loss == 'TverskyLoss':
        kwargs.pop('type')
        loss = smp.losses.TverskyLoss(**kwargs)
    elif loss == 'CrossEntropyLoss':
        kwargs.pop('type')
        if 'weight' in kwargs and isinstance(kwargs['weight'], (list, tuple)):
            kwargs['weight'] = torch.FloatTensor(kwargs['weight'])
        loss = torch.nn.CrossEntropyLoss(**kwargs)
    elif loss == 'BinaryCrossEntropyLoss':
        raise NotImplementedError
        kwargs.pop('type')
        loss = torch.nn.BCELoss(**kwargs)
    elif loss == 'OrderedCrossEntropyLoss':
        from losses import OrderedCrossEntropyLoss
        kwargs.pop('type')
        loss = OrderedCrossEntropyLoss(**kwargs)
    elif loss == 'GCELoss':
        from losses import GCELoss
        kwargs.pop('type')
        if 'weight' in kwargs and isinstance(kwargs['weight'], (list, tuple)):
            kwargs['weight'] = torch.FloatTensor(kwargs['weight'])
        loss = GCELoss(**kwargs)
    elif loss == 'EMDLoss':
        from losses import EMDLoss
        kwargs.pop('type')
        loss = EMDLoss(**kwargs)
    elif loss == 'MatrixExpectedRiskLoss':
        from losses import MatrixExpectedRiskLoss
        kwargs.pop('type')
        loss = MatrixExpectedRiskLoss(cost_matrix=NAVIGATION_COST_MATRIX, **kwargs)
    elif loss == 'CostSensitiveCrossEntropyLoss':
        from losses import CostSensitiveCrossEntropyLoss
        kwargs.pop('type')
        loss = CostSensitiveCrossEntropyLoss(cost_matrix=NAVIGATION_COST_MATRIX, **kwargs)
    elif loss == 'AsymmetricSoftLabelLoss':
        from losses import AsymmetricSoftLabelLoss
        kwargs.pop('type')
        loss = AsymmetricSoftLabelLoss(cost_matrix=NAVIGATION_COST_MATRIX, **kwargs)
    elif loss == 'OrdinalBrierScoreLoss':
        from losses import OrdinalBrierScoreLoss
        kwargs.pop('type')
        loss = OrdinalBrierScoreLoss(cost_matrix=NAVIGATION_COST_MATRIX, **kwargs)
    elif loss == 'MixedLoss':
        from losses import MixedLoss
        kwargs.pop('type')
        losses_cfg = kwargs.pop('losses')
        weights = kwargs.pop('weights', None)
        # Recursively build each constituent loss.
        sub_losses = [get_loss(lc['type'], chart=chart, **dict(lc)) for lc in losses_cfg]
        loss = MixedLoss(sub_losses, weights)
    elif loss == 'CORALLoss':
        from losses import CORALLoss
        kwargs.pop('type')
        loss = CORALLoss(cost_matrix=NAVIGATION_COST_MATRIX, **kwargs)
    elif loss == 'MSELossFromLogits':
        from losses import MSELossFromLogits
        kwargs.pop('type')
        loss = MSELossFromLogits(chart=chart, **kwargs)
    elif loss == 'MSELoss':
        kwargs.pop('type')
        loss = torch.nn.MSELoss(**kwargs)
    elif loss == 'MSELossWithIgnoreIndex':
        from losses import MSELossWithIgnoreIndex
        kwargs.pop('type')
        loss = MSELossWithIgnoreIndex(**kwargs)
    else:
        raise ValueError(f'The given loss \'{loss}\' is unrecognized or Not implemented')

    return loss


def get_model(train_options, device):
    if train_options['model_selection'] in ['dbunet', 'DBUNet']:
        from DBU_Net import DBUNet_ASPP
        net = DBUNet_ASPP(options=train_options).to(device)
    elif train_options['model_selection'] == 'unet':
        net = UNet(options=train_options).to(device)
    elif train_options['model_selection'] == 'swin':
        from swin_transformer import SwinTransformer
        net = SwinTransformer(options=train_options).to(device)
    elif train_options['model_selection'] == 'PA2':
        from PA2_Swin import SwinTransformer
        net = SwinTransformer(options=train_options).to(device)
    elif train_options['model_selection'] == 'PA2_swin_improved':
        from PA2_Swin_Improved import SwinTransformerImproved
        net = SwinTransformerImproved(options=train_options).to(device)
    elif train_options['model_selection'] == 'PA2_swin_improved_old':
        from PA2_Swin_Improved_old import SwinTransformerImproved
        net = SwinTransformerImproved(options=train_options).to(device)
    # 新增 SegNetXt 支持
    elif train_options['model_selection'] == 'SegNetXt':
        from SegNetXt import SegNeXt
        # 根据输入变量数量计算输入通道数
        input_channels = len(train_options['train_variables'])
        # 调整 embed_dims 的第一个维度为实际输入通道数
        embed_dims = [input_channels, 32, 64, 160, 256]
        
        net = SegNeXt(num_classes=train_options['n_classes'], embed_dims=embed_dims).to(device)
        
    elif train_options['model_selection'] == 'h_unet':
        from unet import H_UNet
        net = H_UNet(options=train_options).to(device)
    elif train_options['model_selection'] == 'h_unet_argmax':
        from unet import H_UNet_argmax
        net = H_UNet_argmax(options=train_options).to(device)
    elif train_options['model_selection'] == 'Separate_decoder':
        net = Sep_feat_dif_stages(options=train_options).to(device)
    elif train_options['model_selection'] in ['UNet_regression', 'unet_regression']:
        from unet import UNet_regression
        net = UNet_regression(options=train_options).to(device)
    elif train_options['model_selection'] in ['UNet_regression_all']:
        from unet import UNet_regression_all
        net = UNet_regression_all(options=train_options).to(device)
    elif train_options['model_selection'] in ['UNet_sep_dec_regression', 'unet_sep_dec_regression']:
        from unet import UNet_sep_dec_regression
        net = UNet_sep_dec_regression(options=train_options).to(device)
    elif train_options['model_selection'] in ['UNet_sep_dec_mse']:
        from unet import UNet_sep_dec_mse
        net = UNet_sep_dec_mse(options=train_options).to(device)
    elif train_options['model_selection'] == 'smp':
        from models.smp import SMPWrapper
        net = SMPWrapper(options=train_options).to(device)
    elif train_options['model_selection'] == 'mmseg':
        from models.mmseg import MMSegWrapper
        net = MMSegWrapper(options=train_options).to(device)
    elif train_options['model_selection'] == 'pa2_fpn_mit_b2':
        from PA2_MiT_FPN import PA2FPN_MiT_B2
        net = PA2FPN_MiT_B2(options=train_options).to(device)
    # ------------------------------------------------------------------ #
    # SOTA comparison models
    # ------------------------------------------------------------------ #
    elif train_options['model_selection'] == 'sota_unet':
        from models.sota.unet import UNet as _SOTAUNet
        net = _SOTAUNet(options=train_options).to(device)
    elif train_options['model_selection'] == 'sota_segnet':
        from models.sota.segnet import SegNet as _SegNet
        net = _SegNet(options=train_options).to(device)
    elif train_options['model_selection'] == 'sota_resnet':
        from models.sota.resnet import ResNetFPN
        net = ResNetFPN(options=train_options).to(device)
    elif train_options['model_selection'] == 'sota_densenet':
        from models.sota.densenet import DenseNetFPN
        net = DenseNetFPN(options=train_options).to(device)
    elif train_options['model_selection'] == 'sota_pspnet':
        from models.sota.pspnet import PSPNet as _PSPNet
        net = _PSPNet(options=train_options).to(device)
    elif train_options['model_selection'] == 'sota_deeplabv3':
        from models.sota.deeplabv3 import DeepLabV3Plus
        net = DeepLabV3Plus(options=train_options).to(device)
    elif train_options['model_selection'] == 'sota_segformer':
        from models.sota.segformer import SegFormer
        net = SegFormer(options=train_options).to(device)
    elif train_options['model_selection'] == 'sota_poolformer':
        from models.sota.poolformer import PoolFormerFPN
        net = PoolFormerFPN(options=train_options).to(device)
    elif train_options['model_selection'] == 'sota_segnext':
        from models.sota.segnext import SegNeXt
        net = SegNeXt(options=train_options).to(device)
    else:
        raise 'Unknown model selected'
    return net
