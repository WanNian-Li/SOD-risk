# Dimension-2 experiment: Inverted non-uniform cost matrix
#
# α increases toward high ice (opposite gradient to 'current'):
#   Water→NI=2, NI→TFYI=4, TFYI→ThkFYI=4, ThkFYI→MYI=6
#
# Matrix:
#   [[ 0,  1,  2,  3,  4],
#    [ 2,  0,  1,  2,  3],
#    [ 6,  4,  0,  1,  2],
#    [10,  8,  4,  0,  1],
#    [16, 14, 10,  6,  0]]
#
# C_max=16, avg α=4 — identical to 'moderate' (uniform) and 'current' (non-uniform).
# The only variable is the gradient direction of α across class boundaries:
#   current:  α=6,4,4,2  (high penalty at Water/NI, low at ThkFYI/MYI)
#   inverted: α=2,4,4,6  (low penalty at Water/NI, high at ThkFYI/MYI)
#   moderate: α=4,4,4,4  (uniform, Phase 4 control)
_base_ = ['./PA2_swin_CE_risk.py']

train_options = {
    'cost_matrix_variant': 'inverted',
    'model_save_criterion': 'mean_f1',
    'early_stop_patience': 30,
    'wandb_upload_model': False,
}
