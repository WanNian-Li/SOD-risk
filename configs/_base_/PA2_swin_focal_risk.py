# FocalLoss (gamma=1.5) + risk-optimal decoding + NRS-constrained mean F1 criterion.
# Focal loss auto-upweights hard pixels (New Ice, Thin FYI) without explicit class weights.
# FocalLoss is in src/losses.py: __init__(gamma, weight=None, ignore_index=255)
_base_ = ['./PA2_swin_CE_risk.py']

train_options = {
    'loss': {
        'SOD': {
            'type': 'FocalLoss',
            'gamma': 1.5,
            'ignore_index': 255,
        },
    },
    'model_save_criterion': 'nrs_f1',
    'early_stop_patience': 80,
}
