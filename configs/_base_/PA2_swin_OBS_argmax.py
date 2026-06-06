# Experiment: OBS training + argmax inference (diagnostic)
#
# Purpose: isolate whether the F1 drop (89% → 83%) comes from:
#   (a) OBS training pushing probabilities toward higher ice classes, OR
#   (b) risk-optimal inference biasing decisions toward higher classes, OR
#   (c) both.
#
# By keeping OBS training but switching inference back to argmax,
# we can read off the training-only effect on F1.
#
# Interpretation:
#   - F1(OBS+argmax) ≈ 89%: F1 drop is entirely from risk-optimal inference
#   - F1(OBS+argmax) ≈ 83%: F1 drop is entirely from OBS training
#   - F1(OBS+argmax) between 83%–89%: both training and inference contribute

_base_ = ['./PA2_swin_OBS.py']

train_options = {
    'inference_decision': 'argmax',
}
