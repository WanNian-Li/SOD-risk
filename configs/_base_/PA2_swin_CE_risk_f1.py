# CE_risk training; save checkpoint with highest simple mean F1 (no NRS constraint).
# Diagnostic: reveals the F1 ceiling achievable with risk-optimal decoding.
_base_ = ['./PA2_swin_CE_risk.py']

train_options = {
    'model_save_criterion': 'mean_f1',
}
