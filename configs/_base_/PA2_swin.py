# 用法：
# python quickstart.py configs/SOD/pa2_swin.py --wandb-project MyDS --work-dir ./work_dirs/PA2_Swin_cv --wandb-name PA2_Swin_cv


_base_ = ['./base.py']

train_options = {
    'path_to_train_data': '/root/autodl-tmp/My_dataset/',
    'path_to_test_data': '/root/autodl-tmp/My_dataset/',

    'train_list_path': 'datalists/cv_folds6/fold7_train.json',
    'val_path': 'datalists/cv_folds6/fold7_val_filter.json',
    'test_path': 'datalists/cv_folds6/fold7_val_filter.json',

    'sequential_cv': False,
    'cv_fold_dir': 'datalists/cv_folds6',
    'cv_num_folds': 8,
    'preload_all_cv_scenes': True,

    'enable_scene_cache': True,

    'model_selection': 'PA2',   # ← PA²-Swin (AttnRes + Adaptive Gate + Pol Prior)

    'compute_classwise_f1score': True,
    'plot_confusion_matrix': True,
    'compile_model': True,

    'optimizer': {
        'type': 'AdamW',
        'lr': 1e-4,
        'b1': 0.9,
        'b2': 0.999,
        'eps': 1e-8,
        'weight_decay': 0.05,
    },
    'scheduler': {
        'type': 'CosineAnnealingLR',
        'lr_min': 1e-6,
    },

    'chart_loss': {
        'SIC': {
            'type': 'MSELossFromLogits',
            'ignore_index': 255,
        },
        'SOD': {
            'type': 'MatrixExpectedRiskLoss',
            'ignore_index': 255,
        },
        'FLOE': {
            'type': 'CrossEntropyLoss',
            'ignore_index': 255,
        },
    },
    'target_chart': 'SOD',

    # Ordinal score: monitor and save best model by NRS (Navigation Risk Score).
    # Requires ordinal_metric.enabled=True for OrdinalScore to also be computed.
    'ordinal_metric': {'enabled': True},
    'model_save_criterion': 'nrs',

    # Risk-optimal inference: argmin expected cost under NAVIGATION_COST_MATRIX.
    'inference_decision': 'ordinal_risk_optimal',

    'task_weights': [1],

    'early_stop_patience':20,

    'seed': 10,
    'epochs': 200,
    'epoch_len': 100,

    'num_workers': 18,
    'num_workers_val': 6,
    'prefetch_factor': 4,
    'patch_size': 128,
    'batch_size': 64,
    'down_sample_scale': 5,
    'val_freq': 1,
    'val_downsample_scale': 5,

    'swin_hp': {
        'val_stride': [128, 128],
        'test_stride': [64, 64],
    },

    'patch_log_mode': 'per_epoch',
    # ------------------------------------------------------------------ #
    # 数据过滤
    # ------------------------------------------------------------------ #
    'boundary_erosion_iters': 1,
    'cls2_filter_mask_dir': '',

    'sod_invalid_max_ratio': 0.5,
    'water_patch_max_ratio': 1.0,
    'water_rejection_prob':  0.7,
    'cls4_patch_max_ratio':  1.0,
    'cls4_rejection_prob':   0.0,
    'rare_samplers': [
        {'classes': [2], 'alpha': 0.9},
        {'classes': [1], 'alpha': 0.6},
    ],

    'dynamic_loss_weight':   False,  # EMDLoss has no per-class weight parameter
    'dynamic_weight_warmup': 5,
    'dynamic_weight_ema':    0.7,
    'dynamic_weight_max':    5.0,

    'pol_ratio_channel': True,   # 必须为 True，PA² 的 pol_map 提取依赖此通道

    'data_augmentations': {
        'Random_h_flip': 0.5,
        'Random_v_flip': 0.5,
        'Random_rotation_prob': 0.5,
        'Random_rotation': 90,
        'Random_scale_prob': 0.5,
        'Random_scale': (0.9, 1.1),
        'Cutmix_beta': 1.0,
        'Cutmix_prob': 0.5,
    },
}
