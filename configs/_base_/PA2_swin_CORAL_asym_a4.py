# 方案 2: asymmetric ordinal thresholds — α=4 (boundary p* = 0.2). See _a1 for full notes.
_base_ = ['./PA2_swin_CORAL.py']

train_options = {
    'inference_decision': 'argmax',
    'chart_loss': {
        'SOD': {
            'type': 'CORALLoss',
            'num_classes': 5,
            'ignore_index': 255,
            'under_alpha': 4.0,
        },
    },
}
