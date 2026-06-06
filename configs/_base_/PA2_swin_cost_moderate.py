# Sensitivity analysis: Moderate asymmetry cost matrix (α=4, uniform)
#
# Underestimation cost = 4 × steps, overestimation cost = 1 × steps.
# Uniform 4:1 asymmetry throughout all class boundaries.  C_max = 16.
#
# Matrix:
#   [[ 0,  1,  2,  3,  4],
#    [ 4,  0,  1,  2,  3],
#    [ 8,  4,  0,  1,  2],
#    [12,  8,  4,  0,  1],
#    [16, 12,  8,  4,  0]]
#
# C_max = 16 matches the current matrix, making NRS values directly comparable.
# Unlike the current matrix (non-uniform, 6:1 at Water/Ice down to 2:1 at MYI),
# this version applies the same 4:1 ratio at every boundary.
_base_ = ['./PA2_swin_CE_risk.py']

train_options = {
    'cost_matrix_variant': 'moderate',
    'model_save_criterion': 'mean_f1',
}
