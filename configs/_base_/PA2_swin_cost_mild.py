# Sensitivity analysis: Mild asymmetry cost matrix (α=2)
#
# Underestimation cost = 2 × steps, overestimation cost = 1 × steps.
# Uniform 2:1 asymmetry throughout all class boundaries.  C_max = 8.
#
# Matrix:
#   [[ 0,  1,  2,  3,  4],
#    [ 2,  0,  1,  2,  3],
#    [ 4,  2,  0,  1,  2],
#    [ 6,  4,  2,  0,  1],
#    [ 8,  6,  4,  2,  0]]
#
# Compared to current (α≈4–6, non-uniform): tests whether the current matrix
# over-penalises underestimation, causing unnecessary F1 loss.
_base_ = ['./PA2_swin_CE_risk.py']

train_options = {
    'cost_matrix_variant': 'mild',
    'model_save_criterion': 'mean_f1',
}
