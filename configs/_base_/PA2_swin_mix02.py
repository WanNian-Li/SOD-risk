# Experiment: Mixed loss CE + 0.2·OBS, risk-optimal inference
#
# Loss: L = 1.0 * CrossEntropyLoss + 0.2 * OrdinalBrierScoreLoss
#
# Intuition:
#   CE dominates (weight=1.0) → classification accuracy / F1 preserved
#   OBS provides mild ordinal risk gradient (weight=0.2) → NRS nudged upward
#
# Scale note: at a well-trained model (F1~89%), CE ≈ 0.5 and OBS ≈ 0.1,
# so OBS contributes roughly 2-5% of the total gradient signal.
# This is intentionally gentle to test whether any risk signal is needed
# during training on top of risk-optimal inference.
#
# Expected outcome: F1 ≈ 88-89%, NRS > CE+risk (best of both worlds)

_base_ = ['./PA2_swin.py']

train_options = {
    'chart_loss': {
        'SOD': {
            'type': 'MixedLoss',
            'losses': [
                {'type': 'CrossEntropyLoss', 'ignore_index': 255},
                {'type': 'OrdinalBrierScoreLoss', 'num_classes': 5, 'ignore_index': 255},
            ],
            'weights': [1.0, 0.2],
        },
    },
    # inference_decision: 'ordinal_risk_optimal' inherited from PA2_swin.py
    # model_save_criterion: 'nrs' inherited from PA2_swin.py
}
