# Smoke test for cost matrix sensitivity experiments.
# Runs cost_mild (the simplest variant) for 2 epochs to verify:
#   1. select_cost_matrix() correctly replaces the global matrix
#   2. nrs_ref_scores (current matrix reference) is computed without errors
#   3. mean_f1 criterion works with the new code path
_base_ = ['./PA2_swin_cost_mild.py']

train_options = {
    'train_list_path': 'datalists/cv_folds6/fold6_train.json',
    'epochs': 2,
    'early_stop_patience': 999,
}
