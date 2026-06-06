# Experiment: CE training + risk-optimal inference
#
# Training: CrossEntropyLoss (same as the 89% F1 baseline)
# Inference: ordinal_risk_optimal argmin under NAVIGATION_COST_MATRIX
#
# Hypothesis: training stays CE-optimal (F1 preserved), but at inference
# the cost-matrix guides the decision boundary → NRS improves "for free"
# with zero change to training.
#
# Key paper contribution C4 candidate:
#   If NRS(CE+risk) > NRS(CE+argmax) with F1 ≈ same,
#   the inference-only change alone is a valid contribution.

_base_ = ['./PA2_swin.py']

train_options = {
    'chart_loss': {
        'SOD': {
            'type': 'CrossEntropyLoss',
            'ignore_index': 255,
        },
    },
    # inference_decision: 'ordinal_risk_optimal' and model_save_criterion: 'nrs'
    # are already set in PA2_swin.py.
}
