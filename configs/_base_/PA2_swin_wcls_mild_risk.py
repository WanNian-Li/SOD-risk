# Class-weighted CrossEntropy + risk-optimal decoding + NRS-constrained mean F1 criterion.
# Weights: inverse class frequency (val proportions: Water=23.3%, NI=12.4%, TFYI=10.5%, ThkFYI=24.2%, MYI=29.6%),
# normalized to mean=1.0: [0.70, 1.40, 1.60, 0.70, 0.60]
_base_ = ['./PA2_swin_CE_risk.py']

train_options = {
    'loss': {
        'SOD': {
            'type': 'CrossEntropyLoss',
            'ignore_index': 255,
            'weight': [0.70, 1.40, 1.60, 0.70, 0.60],
        },
    },
    'model_save_criterion': 'nrs_f1',
    'early_stop_patience': 80,
}
