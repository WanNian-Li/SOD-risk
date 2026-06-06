# Sensitivity analysis: Moderate-v3 cost matrix
#
# Lower triangle: same as moderate — underestimation cost = 4 × steps
# Upper triangle: C_upper(d) = 2(d-1) + 1  →  1, 3, 5, 7  for d = 1, 2, 3, 4
#   Adjacent overestimation stays cheap (cost=1);
#   each extra step adds 2, making large overestimation progressively expensive.
#
# Ratio lower:upper narrows with distance (4:1 → 2.3:1), unlike moderate
# where the ratio is constant 4:1 at every distance.
# C_max = 16 (same as moderate → NRS directly comparable).
#
# Matrix:
#   [[ 0,  1,  3,  5,  7],
#    [ 4,  0,  1,  3,  5],
#    [ 8,  4,  0,  1,  3],
#    [12,  8,  4,  0,  1],
#    [16, 12,  8,  4,  0]]
#
# Analogue of severe_v3 at moderate scale.
# Question: does penalising large overestimation (while keeping adjacent
# overestimation cheap) change OVR / UR compared to pure moderate?

_base_ = ['./PA2_swin_CE_risk.py']

train_options = {
    'cost_matrix_variant': 'moderate_v3',
    'model_save_criterion': 'mean_f1',
    'plot_cost_weighted_confusion_matrix': True,
}
