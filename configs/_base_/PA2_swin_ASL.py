# Experiment: Asymmetric Soft Label Loss (ASL)
#
# Baseline comparison against PA2_swin.py (MERL).
#
# Key difference: instead of minimising expected cost directly (MERL),
# we change *what the model is asked to output*.  For true class c the
# training target is the Boltzmann distribution over prediction classes:
#
#     q_j = softmax(-temperature * C[c, j])
#
# High-cost predictions receive near-zero target probability; the correct
# class (C[c,c]=0) receives the highest.  The model is trained with soft
# cross-entropy H(q, p) instead of one-hot CE.
#
# MERL vs ASL:
#   MERL  — changes the gradient weight (expected cost descent)
#   ASL   — changes the target distribution (cost-weighted soft label)
# They are orthogonal; both use the same NAVIGATION_COST_MATRIX.
#
# Usage:
#   python scripts/run_batch.py \
#       configs/_base_/PA2_swin.py configs/_base_/PA2_swin_ASL.py \
#       --wandb-project MyDS --work-dir work_dirs/loss_ablation

_base_ = ['./PA2_swin.py']

train_options = {
    'chart_loss': {
        'SOD': {
            'type': 'AsymmetricSoftLabelLoss',
            # temperature controls label softness:
            #   > 1  → sharper targets (closer to one-hot CE)
            #   = 1  → cost-weighted soft labels (default)
            #   < 1  → very soft, more mass on adjacent classes
            'temperature': 1.0,
            'ignore_index': 255,
        },
    },
}
