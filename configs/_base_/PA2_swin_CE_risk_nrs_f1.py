# CE_risk training; save checkpoint that maximizes simple mean F1 subject to NRS > 0.9808 (CE_argmax baseline)
_base_ = ['./PA2_swin_CE_risk.py']

train_options = {
    'model_save_criterion': 'nrs_f1',
    'early_stop_patience': 80,
}
