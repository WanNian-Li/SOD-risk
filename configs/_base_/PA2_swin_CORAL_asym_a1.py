# 方案 2: risk-aware asymmetric ordinal thresholds (CORAL + under_alpha).
#
# under_alpha = α scales ONLY the underestimation BCE term t_k·(-log p_k), which
# shifts every ordinal decision boundary from p_k = 0.5 to p_k* = 1/(1+α),
# training a risk-averse (prefer-higher-ice) classifier intrinsically.
#
# Decode is NEUTRAL argmax: risk preference lives in TRAINING, not in decoding,
# so we do NOT stack expected-cost / CVaR decoding on top (avoids the
# double-counting that made decode-time CVaR fail).  The cost matrix is held
# fixed at 'current' (for CORAL w_k and for NRS), so this sweep isolates α only.
#
# a1 = symmetric baseline (α=1, p*=0.5) + neutral decode — the control arm.
_base_ = ['./PA2_swin_CORAL.py']

train_options = {
    'inference_decision': 'argmax',
    'chart_loss': {
        'SOD': {
            'type': 'CORALLoss',
            'num_classes': 5,
            'ignore_index': 255,
            'under_alpha': 1.0,
        },
    },
}
