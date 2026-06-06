# Sensitivity analysis: Severe asymmetry cost matrix (α=8)
#
# Underestimation cost = 8 × steps, overestimation cost = 1 × steps.
# Uniform 8:1 asymmetry throughout all class boundaries.  C_max = 32.
#
# Matrix:
#   [[ 0,  1,  2,  3,  4],
#    [ 8,  0,  1,  2,  3],
#    [16,  8,  0,  1,  2],
#    [24, 16,  8,  0,  1],
#    [32, 24, 16,  8,  0]]
#
# Tests the extreme end: heavy punishment for any underestimation.
# Expected: highest NRS improvement but most F1 degradation.
_base_ = ['./PA2_swin_CE_risk.py']

train_options = {
    'cost_matrix_variant': 'severe',
    'model_save_criterion': 'mean_f1',
    'early_stop_patience': 30,
    'wandb_upload_model': False,
}
