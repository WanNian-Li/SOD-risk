#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Batch experiment runner: load training data ONCE, run multiple configs sequentially.

The global _SCENE_CACHE in loaders.py survives across training runs as long as the
Python process stays alive.  This script exploits that: all scenes required by
every config are preloaded into the cache before the first training job starts,
so subsequent configs pay zero I/O cost for scenes they share.

Configs are grouped by their "cache key" (scale / patch_size / variables / …).
Scenes are preloaded once per group.  If two configs have different keys they form
separate groups and each group gets its own preload — you still save the reload
cost for configs within the same group.

Usage
-----
python run_batch.py \\
    configs/SOD/all.py configs/SOD/smp_fpn_mit_b2.py \\
    --wandb-project MyDS \\
    [--wandb-names name_all name_fpn_mit_b2] \\
    [--work-dir work_dirs/batch_run] \\
    [--seed 42]

    
python run_batch.py configs/SOTA/unet.py configs/SOTA/segnext.py configs/SOTA/segnet.py configs/SOTA/segformer.py configs/SOTA/resnet.py configs/SOTA/pspnet.py configs/SOTA/poolformer.py configs/SOTA/densenet.py configs/SOTA/deeplabv3.py --wandb-project MyDS --work-dir work_dirs/SOTA_comparison 




Arguments
---------
configs         One or more config file paths (processed in order).
--wandb-project W&B project name (required).
--wandb-names   W&B run names, one per config.  Defaults to config filename stems.
--work-dir      Base work directory.  Each config gets <work-dir>/<config_stem>/.
                Falls back to <cfg.work_dir> then ./work_dirs/<config_stem>/.
--seed          Override the random seed for ALL configs.
"""

import argparse
import copy
import json
import os
import os.path as osp
import pathlib
import re
import shutil
import warnings
from collections import defaultdict
from types import SimpleNamespace

warnings.filterwarnings("ignore")

import numpy as np
import torch
from mmcv import Config, mkdir_or_exist

from src.loaders import get_variable_options, preload_scene_cache
from quickstart import (
    _collect_cv_folds,
    _read_json_list,
    _run_training_job,
    _set_global_seed,
    _setup_device,
)


# ------------------------------------------------------------------ #
# Cache-group helpers                                                  #
# ------------------------------------------------------------------ #

def _cache_group_key(options):
    """Hashable key: two configs with the same key share the scene cache."""
    return (
        options['path_to_train_data'],
        options['down_sample_scale'],
        options['loader_downsampling'],
        options['patch_size'],
        tuple(options['full_variables']),
        tuple(options['charts']),
        options.get('cls2_filter_mask_dir', None),   # keep None vs '' distinct
    )


def _collect_all_train_files(options):
    """Return every training .nc filename referenced by this config.

    Handles both sequential-CV configs (reads all fold_N_train.json files) and
    plain configs (reads train_list_path).
    """
    path_to_env = options.get('path_to_env', './')
    files = set()

    if options.get('sequential_cv', False):
        fold_dir = options.get('cv_fold_dir', None)
        if fold_dir:
            fold_dir_rel = fold_dir.replace('\\', '/').rstrip('/')
            fold_dir_abs = osp.join(path_to_env, fold_dir_rel)
            if osp.isdir(fold_dir_abs):
                train_pat = re.compile(r'^fold(\d+)_train\.json$')
                fold_ids = []
                for name in os.listdir(fold_dir_abs):
                    m = train_pat.match(name)
                    if m:
                        fold_ids.append(int(m.group(1)))
                fold_ids = sorted(set(fold_ids))

                requested = int(options.get('cv_num_folds', 0) or 0)
                if requested > 0:
                    fold_ids = fold_ids[:requested]

                for fid in fold_ids:
                    p = osp.join(path_to_env, fold_dir_rel, f'fold{fid}_train.json')
                    if osp.exists(p):
                        files.update(_read_json_list(p))

    # Always also include the default train_list_path (covers non-CV configs and
    # provides a fallback for CV configs whose fold dir is missing).
    train_path = osp.join(path_to_env, options.get('train_list_path', ''))
    if osp.exists(train_path):
        files.update(_read_json_list(train_path))

    return sorted(files)


# ------------------------------------------------------------------ #
# Argument parsing                                                     #
# ------------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run multiple experiment configs sequentially, loading data once.')
    parser.add_argument(
        'configs', nargs='+', type=pathlib.Path,
        help='Config file paths (processed in order)')
    parser.add_argument(
        '--wandb-project', required=True,
        help='W&B project name')
    parser.add_argument(
        '--wandb-names', nargs='*', default=None,
        help='W&B run names (one per config).  Defaults to config filename stems.')
    parser.add_argument(
        '--work-dir', default=None,
        help='Base work directory; each config gets a sub-directory '
             '<work-dir>/<config_stem>/')
    parser.add_argument(
        '--seed', default=None, type=int,
        help='Override the random seed for ALL configs')
    return parser.parse_args()


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main():
    args = parse_args()

    if args.wandb_names is not None and len(args.wandb_names) != len(args.configs):
        raise ValueError(
            f'--wandb-names has {len(args.wandb_names)} entries but '
            f'{len(args.configs)} config(s) were provided.')

    wandb_names = args.wandb_names or [p.stem for p in args.configs]

    # ------------------------------------------------------------------ #
    # Step 1 — Load every config and collect metadata                     #
    # ------------------------------------------------------------------ #
    experiments = []
    for cfg_path, wname in zip(args.configs, wandb_names):
        cfg = Config.fromfile(cfg_path)
        options = copy.deepcopy(cfg.train_options)
        options = get_variable_options(options)   # populates full_variables etc.

        if args.seed is not None:
            options['seed'] = args.seed

        train_files = _collect_all_train_files(options)
        key = _cache_group_key(options)

        experiments.append({
            'cfg_path': cfg_path,
            'cfg': cfg,
            'options': options,
            'wandb_name': wname,
            'cache_key': key,
            'train_files': train_files,
        })
        print(f'[Loaded] {cfg_path.name}: '
              f'{len(train_files)} training scenes | '
              f'scale={options["down_sample_scale"]}, '
              f'patch={options["patch_size"]}')

    # ------------------------------------------------------------------ #
    # Step 2 — Group configs by cache key                                 #
    # ------------------------------------------------------------------ #
    # Use an ordered dict so configs run in the order they were specified.
    groups = defaultdict(list)
    for exp in experiments:
        groups[exp['cache_key']].append(exp)

    if len(groups) > 1:
        print(
            f'\n[INFO] Configs span {len(groups)} cache-key groups '
            f'(different scale / patch_size / variables). '
            f'Scenes will be preloaded once per group — '
            f'no cross-group cache sharing.\n')

    # ------------------------------------------------------------------ #
    # Step 3 — For each group: one-time preload → run configs in order    #
    # ------------------------------------------------------------------ #
    total_cfgs = len(experiments)
    run_counter = 0

    for g_idx, (cache_key, group_exps) in enumerate(groups.items()):
        ref_options = group_exps[0]['options']
        all_files = sorted({f for e in group_exps for f in e['train_files']})

        print(f'\n{"="*64}')
        print(f'Cache group {g_idx + 1}/{len(groups)}: '
              f'{len(group_exps)} config(s), '
              f'{len(all_files)} unique training scene(s)')
        print(f'  scale={ref_options["down_sample_scale"]}, '
              f'patch_size={ref_options["patch_size"]}, '
              f'charts={ref_options["charts"]}')
        print('='*64)

        # One-time preload for this cache group.
        if ref_options.get('enable_scene_cache', True) and all_files:
            print(f'\nPreloading {len(all_files)} scenes into RAM '
                  f'(one-time cost for this group)...')
            preload_scene_cache(ref_options, all_files)
            print('Preload complete.  Subsequent Dataset inits will be cache hits.\n')
        else:
            print('[SceneCache] Cache disabled or no training files — skipping preload.')

        # Run each config in this group sequentially.
        for exp in group_exps:
            run_counter += 1
            cfg_path = exp['cfg_path']
            cfg      = exp['cfg']
            options  = exp['options']
            wname    = exp['wandb_name']

            print(f'\n{"─"*64}')
            print(f'[{run_counter}/{total_cfgs}]  {cfg_path.name}  |  wandb: {wname}')
            print(f'{"─"*64}')

            # Random seed.
            if options['seed'] != -1:
                _set_global_seed(options['seed'])
                print(f'Seed: {options["seed"]}')
            else:
                print('Random seed chosen.')

            # Determine work directory.
            if args.work_dir is not None:
                base_work_dir = osp.join(args.work_dir, cfg_path.stem)
            elif cfg.get('work_dir', None) is not None:
                base_work_dir = cfg.work_dir
            else:
                base_work_dir = osp.join('./work_dirs', cfg_path.stem)

            mkdir_or_exist(osp.abspath(base_work_dir))
            shutil.copy(cfg_path, osp.join(base_work_dir, cfg_path.name))

            # Minimal args namespace expected by _run_training_job.
            job_args = SimpleNamespace(
                config=cfg_path,
                wandb_project=args.wandb_project,
                wandb_name=wname,
                resume_from=None,
                finetune_from=None,
            )

            # GPU device (resolved per config so gpu_id can differ).
            device = _setup_device(options)

            if options.get('sequential_cv', False):
                # ---- Sequential k-fold CV ----------------------------------------
                folds = _collect_cv_folds(options)
                if not folds:
                    raise ValueError(
                        f'sequential_cv=True but no folds found for {cfg_path.name}. '
                        f'Check train_options["cv_fold_dir"].')

                print(f'Sequential CV mode: {len(folds)} fold(s)')

                for i, fold in enumerate(folds):
                    fold_id = fold['fold']
                    print(f'\n--- Fold {fold_id}  ({i + 1}/{len(folds)}) ---')

                    fold_cfg = cfg.deepcopy()
                    fold_cfg.work_dir = osp.join(base_work_dir, f'fold{fold_id}')
                    mkdir_or_exist(osp.abspath(fold_cfg.work_dir))
                    shutil.copy(cfg_path, osp.join(fold_cfg.work_dir, cfg_path.name))

                    fold_options = copy.deepcopy(options)
                    fold_options['train_list_path'] = fold['train_list_path']
                    fold_options['val_path'] = fold['val_path']

                    _run_training_job(
                        fold_cfg, job_args, fold_options, device, fold_tag=fold_id)

                    print(f'--- Fold {fold_id} complete ---')

            else:
                # ---- Single run --------------------------------------------------
                cfg.work_dir = base_work_dir
                _run_training_job(cfg, job_args, options, device)

            print(f'\n[Done] {cfg_path.name}')

    print(f'\n{"="*64}')
    print(f'All {total_cfgs} experiment(s) complete.')
    print('='*64)


if __name__ == '__main__':
    main()
