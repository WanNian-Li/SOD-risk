# Sensitivity analysis: Severe-v3 cost matrix
#
# Lower triangle: same as severe — underestimation cost = 8 × steps
# Upper triangle: C_upper(d) = 4d - 3  →  1, 5, 9, 13  for d = 1, 2, 3, 4
#   Adjacent overestimation stays cheap (cost=1);
#   each extra step adds 4, making large overestimation progressively expensive.
#
# Ratio lower:upper narrows with distance (8:1 → 2.5:1), unlike existing
# matrices where the ratio is constant at every distance.
# C_max = 32 (same as severe → NRS directly comparable).
#
# Matrix:
#   [[ 0,  1,  5,  9, 13],
#    [ 8,  0,  1,  5,  9],
#    [16,  8,  0,  1,  5],
#    [24, 16,  8,  0,  1],
#    [32, 24, 16,  8,  0]]
#
# Question: does penalising large overestimation (while keeping adjacent
# overestimation cheap) change OVR / UR compared to pure severe?

_base_ = ['./PA2_swin_CE_risk.py']

train_options = {
    'cost_matrix_variant': 'severe_v3',
    'model_save_criterion': 'mean_f1',
    'plot_cost_weighted_confusion_matrix': True,
}
