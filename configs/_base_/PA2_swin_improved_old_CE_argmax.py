# CE_argmax with PA2_Swin_Improved_old (baseline Improved version):
#   - Content-only adaptive gate (no polarization modulation)
#   - GlobalContextInjection without pol-guided query
# Compared to PA2_swin_improved_CE_argmax.py which adds pol modulation.
_base_ = ['./PA2_swin_CE_argmax.py']

train_options = {
    'model_selection': 'PA2_swin_improved_old',
    'early_stop_patience': 30,
}
