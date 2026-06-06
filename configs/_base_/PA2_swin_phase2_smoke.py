# Smoke test for phase2 experiments: 2 epochs to verify all 4 configs work end-to-end.
_base_ = ['./PA2_swin_CE_risk_nrs_f1.py']

train_options = {
    'train_list_path': 'datalists/cv_folds6/fold6_train.json',
    'epochs': 2,
    'early_stop_patience': 999,
}
