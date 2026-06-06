__author__ = 'Muhammed Patel'
__contributor__ = 'Xinwwei chen, Fernando Pena Cantu,Javier Turnes, Eddie Park'
__copyright__ = ['university of waterloo']
__contact__ = ['m32patel@uwaterloo.ca', 'xinweic@uwaterloo.ca']
__version__ = '1.0.0'
__date__ = '2024-04-05'

# -- Built-in modules -- #
import argparse
import json
import random
import os
import os.path as osp
import shutil
from icecream import ic
import pathlib

import numpy as np
import torch
from mmcv import Config, mkdir_or_exist
from tqdm import tqdm  # Progress bar

import wandb
# Functions to calculate metrics and show the relevant chart colorbar.
from src.functions import compute_metrics, save_best_model, load_model, slide_inference, \
    batched_slide_inference, water_edge_metric, class_decider

# Load consutme loss function
from src.losses import WaterConsistencyLoss
# Custom dataloaders for regular training and validation.
from src.loaders import (AI4ArcticChallengeDataset, AI4ArcticChallengeTestDataset,
                         get_variable_options)
#  get_variable_options
from src.unet import UNet, Sep_feat_dif_stages  # Convolutional Neural Network model
from swin_transformer import SwinTransformer  # Swin Transformer
# -- Built-in modules -- #
from src.utils import colour_str
from src.test_upload_function import test
import segmentation_models_pytorch as smp

# python inference.py configs/SOD/all.py work_dirs/My_DS_cls2filter_cv_fold_2_kimi/best_model_My_DS_cls2filter_cv_fold_2_kimi.pth --wandb-project MyDS --work-dir ./work_dirs/My_DS_cls2filter_cv_fold_2_kimi/apply  --mode apply --scene-list-file datalists/test_apply.json
def parse_args():
    parser = argparse.ArgumentParser(description='Train Default U-NET segmentor')

    # Mandatory arguments
    parser.add_argument('config', type=pathlib.Path, help='train config file path',)
    parser.add_argument('checkpoint', type=pathlib.Path, help='checkpoint path of the model',)
    parser.add_argument('--wandb-project', required=True, help='Name of wandb project')
    parser.add_argument('--wandb-disabled', action='store_true',
                        help="完全关闭 wandb 记录（本地推理无需上传时推荐）")
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--mode', choices=['val', 'test', 'apply'], default='apply',
                        help="'apply': 真实推理（无标签，用 4b_pack_for_inference.py 的 NC）；"
                             "'test'/'val': 带 GT 的评估")
    parser.add_argument('--scene-list-key', default=None,
                        help="配置中存放 JSON 路径的 key；默认 apply/test→'test_path'、val→'val_path'")
    parser.add_argument('--scene-list-file', default=None,
                        help="直接传入场景列表 JSON 文件路径，优先级高于 --scene-list-key")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint
    ic(args.config)
    cfg = Config.fromfile(args.config)
    train_options = cfg.train_options
    # Get options for variables, amsrenv grid, cropping and upsampling.
    train_options = get_variable_options(train_options)
    # generate wandb run id, to be used to link the run with test_upload
    id = wandb.util.generate_id()

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        if not train_options['cross_val_run']:
            cfg.work_dir = osp.join('./work_dir',
                                    osp.splitext(osp.basename(args.config))[0])
        else:
            # from utils import run_names
            run_name = id
            cfg.work_dir = osp.join('./work_dir',
                                    osp.splitext(osp.basename(args.config))[0], run_name)

    ic(cfg.work_dir)
    # create work_dir
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    shutil.copy(args.config, osp.join(cfg.work_dir, osp.basename(args.config)))
    cfg_path = osp.join(cfg.work_dir, osp.basename(args.config))
    # ### CUDA / GPU Setup
    # Get GPU resources.
    if torch.cuda.is_available():
        print(colour_str('GPU available!', 'green'))
        print('Total number of available devices: ',
              colour_str(torch.cuda.device_count(), 'orange'))
        device = torch.device(f"cuda:{train_options['gpu_id']}")

    else:
        print(colour_str('GPU not available.', 'red'))
        device = torch.device('cpu')
    print('GPU setup completed!')

    if train_options['model_selection'] == 'unet':
        net = UNet(options=train_options).to(device)
    elif train_options['model_selection'] == 'swin':
        net = SwinTransformer(options=train_options).to(device)
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
    else:
        raise 'Unknown model selected'

    _wandb_kwargs = dict(
        name=osp.splitext(osp.basename(args.config))[0] + '_inference',
        project=args.wandb_project,
        config=train_options,
        id=id,
        resume="allow",
    )
    if args.wandb_disabled:
        _wandb_kwargs['mode'] = 'disabled'
    wandb.init(**_wandb_kwargs)

    # 解析场景列表：优先 --scene-list-file，其次从配置里按 key 取 JSON 路径
    if args.scene_list_file is not None:
        list_json_path = args.scene_list_file
    else:
        _default_list_key = {
            'apply': 'test_path',
            'test':  'test_path',
            'val':   'val_path',
        }
        list_key = args.scene_list_key or _default_list_key[args.mode]
        if list_key not in train_options:
            raise KeyError(f"配置中缺少场景列表 key '{list_key}'（mode={args.mode}）")
        list_json_path = train_options[list_key]

    # 若已是列表（极少数情况下配置直接塞了 list），跳过 JSON 读取
    if isinstance(list_json_path, (list, tuple)):
        scene_list = list(list_json_path)
    else:
        # 先尝试 path_to_env 前缀，再尝试原样路径
        _env = train_options.get('path_to_env', '') or ''
        _candidates = [osp.join(_env, list_json_path), list_json_path]
        for _p in _candidates:
            if osp.isfile(_p):
                with open(_p, 'r') as _f:
                    scene_list = json.loads(_f.read())
                print(f"Loaded {len(scene_list)} scene(s) from {_p}")
                break
        else:
            raise FileNotFoundError(
                f"无法定位场景列表 JSON，尝试过: {_candidates}")

    test(args.mode, net, checkpoint_path, device, cfg,
         scene_list, args.mode)

    # finish the wandb run
    wandb.finish()


if __name__ == '__main__':
    main()
