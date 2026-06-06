# Smoke test for CORAL head: 1 epoch with fold6_train.json (small dataset).
_base_ = ['./PA2_swin_CORAL.py']

train_options = {
    'train_list_path': 'datalists/cv_folds6/fold6_train.json',
    'epochs': 1,
    'early_stop_patience': 999,
}
