# Experiment: Mixed loss CE + 0.5·OBS, risk-optimal inference
#
# Loss: L = 1.0 * CrossEntropyLoss + 0.5 * OrdinalBrierScoreLoss
#
# Intuition: same as mix02 but with a stronger OBS signal.
# At a well-trained model, OBS contributes roughly 8-15% of total gradient.
# This is a medium-strength experiment to find the λ sweet spot.
#
# If mix02 already preserves F1 ≈ 89%, try this for higher NRS.
# If mix05 still preserves F1 ≈ 87-89%, that's the operational range.
#
# Expected outcome: F1 ≈ 86-89%, NRS ≥ mix02

_base_ = ['./PA2_swin.py']

train_options = {
    'chart_loss': {
        'SOD': {
            'type': 'MixedLoss',
            'losses': [
                {'type': 'CrossEntropyLoss', 'ignore_index': 255},
                {'type': 'OrdinalBrierScoreLoss', 'num_classes': 5, 'ignore_index': 255},
            ],
            'weights': [1.0, 0.5],
        },
    },
    # inference_decision: 'ordinal_risk_optimal' inherited from PA2_swin.py
    # model_save_criterion: 'nrs' inherited from PA2_swin.py
}
