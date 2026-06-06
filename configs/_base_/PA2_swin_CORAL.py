# Experiment: CORAL (COnsistent RAnk Logits) ordinal output head
#
# Replaces the standard 5-class softmax head with a shared-weight CORAL head
# that outputs K-1=4 sigmoid values representing P(y > k) for k=0,...,3.
#
# Key architectural change from all previous experiments:
#   - Previous: Linear(embed_dim, 5) → softmax → 5 independent class logits
#   - CORAL:    Linear(embed_dim, 1) + 4 bias params → sigmoid → 4 CDF values
#
# The shared feature axis encodes the ordinal structure directly in the architecture:
#   - All threshold decisions share one underlying "ice severity" score
#   - Biases b[0] > b[1] > b[2] > b[3] naturally maintained by CORALLoss
#   - Prevents the non-monotonic probability distributions that drive OVR=3.83%
#
# Loss: CORALLoss — weighted BCE on each threshold k
#   L = sum_k w_k * BCE(P(y > k), 1[y > k])
#   Threshold weights w_k derived from NAVIGATION_COST_MATRIX (same as OBS).
#
# Inference: coral_cdf_to_probs() converts CDF output to K class probs,
#   then ordinal_risk_optimal argmin selects the minimum-expected-cost class.
#
# Comparison baseline: PA2_swin_OBS.py (OBS — L2 CDF loss, standard 5-class head)
# Target improvement: OVR ↓, Class-1 F1 ↑, NRS ↑ vs OBS baseline of 0.9835

_base_ = ['./PA2_swin.py']

train_options = {
    'sod_head': 'coral',
    'chart_loss': {
        'SOD': {
            'type': 'CORALLoss',
            'num_classes': 5,
            'ignore_index': 255,
        },
    },
}
