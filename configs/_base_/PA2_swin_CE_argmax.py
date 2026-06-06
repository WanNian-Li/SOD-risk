# Experiment: CE + argmax (true F1 / NRS baseline)
#
# This is the reference point for F1-NRS comparison:
#   - Training: standard CrossEntropyLoss (no ordinal structure)
#   - Inference: argmax (no cost-matrix guidance)
#
# Expected: highest F1 (~89%), lowest NRS among all experiments.
# Provides the NRS floor that risk-aware methods should exceed.

_base_ = ['./PA2_swin.py']

train_options = {
    'chart_loss': {
        'SOD': {
            'type': 'CrossEntropyLoss',
            'ignore_index': 255,
        },
    },
    'inference_decision': 'argmax',
    # Save by NRS so the saved epoch is comparable to all other experiments.
    'model_save_criterion': 'nrs',
}
