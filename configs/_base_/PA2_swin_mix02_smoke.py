# Smoke test for MixedLoss (CE + 0.2·OBS) — 2 epochs, small dataset
_base_ = ['./PA2_swin_mix02.py']

train_options = {
    'train_list_path': 'datalists/cv_folds6/fold6_train.json',
    'epochs': 2,
    'early_stop_patience': 999,
}
