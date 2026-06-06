# Smoke test: 1 epoch with a tiny training set to verify ASL config works end-to-end.
_base_ = ['./PA2_swin_ASL.py']

train_options = {
    'train_list_path': 'datalists/cv_folds6/fold6_train.json',
    'epochs': 1,
    'early_stop_patience': 999,
}
