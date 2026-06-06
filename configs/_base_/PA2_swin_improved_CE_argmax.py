# CE_argmax with PA2_Swin_Improved (new version):
#   - Polarization-modulated adaptive gate (pol_stat + pol_std in gate input)
#   - Polarization-guided query in GlobalContextInjection
# Compared to PA2_swin_improved_old_CE_argmax.py which uses the version without pol modulation.
_base_ = ['./PA2_swin_CE_argmax.py']

train_options = {
    'model_selection': 'PA2_swin_improved',
    'early_stop_patience': 30,
}
