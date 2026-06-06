# Experiment: Ordinal Brier Score Loss (OBS)
#
# Baseline comparison against PA2_swin.py (MERL) and PA2_swin_ASL.py (ASL).
#
# The OBS loss is the L2 analogue of EMDLoss (which uses L1 CDF distance):
#
#     L = sum_{k=0}^{K-2} w_k * (CDF_pred(k) - 1[y <= k])^2
#
# Boundary weights w_k are derived from NAVIGATION_COST_MATRIX:
#   w_k = normalised average cost of ALL errors that cross threshold k
#         (both underestimation and overestimation directions).
#
# For the 5-class SOD problem, approximate computed weights:
#   threshold 0 (water|ice):          w_0 ≈ 1.16  (highest — most dangerous)
#   threshold 1 (new|thin FYI):       w_1 ≈ 1.02
#   threshold 2 (thin|thick FYI):     w_2 ≈ 0.93
#   threshold 3 (thick FYI|MYI):      w_3 ≈ 0.88  (lowest)
#
# Theoretical motivation:
#   OBS is a *strictly proper scoring rule* for ordinal distributions (Epstein
#   1969) — its unique minimiser is the true conditional distribution.
#   EMDLoss (L1 CDF) is not a proper scoring rule.  MERL (expected cost) is
#   also proper under the given cost matrix but in a different sense.
#
# Key difference from MERL:
#   MERL minimises expected navigation cost directly (soft argmin).
#   OBS minimises a calibrated CDF fitting objective (soft CDF matching).
#   Both are ordinal and asymmetric; their gradients differ structurally.
#
# Usage:
#   python scripts/run_batch.py \
#       configs/_base_/PA2_swin.py \
#       configs/_base_/PA2_swin_ASL.py \
#       configs/_base_/PA2_swin_OBS.py \
#       --wandb-project MyDS --work-dir work_dirs/loss_ablation

_base_ = ['./PA2_swin.py']

train_options = {
    'chart_loss': {
        'SOD': {
            'type': 'OrdinalBrierScoreLoss',
            'num_classes': 5,
            'ignore_index': 255,
            # cost_matrix is injected automatically from NAVIGATION_COST_MATRIX
            # in get_loss(); no need to specify it here.
        },
    },
}
