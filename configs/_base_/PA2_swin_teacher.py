# 老师建议方案完整实现
#
# ① 惩罚矩阵 W     — NAVIGATION_COST_MATRIX  定义于 src/functions.py:38
# ② CSCS 损失      — CostSensitiveCrossEntropyLoss
#                    L = sum_j C'[c][j] * (-log p_j)，C'对角线置1
# ③ Expected Cost  — model_save_criterion='nrs'（NRS = 1 - EC/C_max）
# ④ 贝叶斯最小风险  — inference_decision='ordinal_risk_optimal'
#                    pred = argmin_j  sum_k p_k * C[k][j]
# ⑤ 代价加权CM对比  — plot_cost_weighted_confusion_matrix=True

_base_ = ['./PA2_swin.py']

train_options = {
    # ② CSCS 替换默认的 MatrixExpectedRiskLoss
    'chart_loss': {
        'SOD': {
            'type': 'CostSensitiveCrossEntropyLoss',
            'ignore_index': 255,
        },
    },

    # ③ Expected Cost 早停（NRS = 1 - EC / C_max，已在 PA2_swin.py 中设置）
    'model_save_criterion': 'nrs',

    # ④ 贝叶斯最小风险推理（已在 PA2_swin.py 中设置）
    'inference_decision': 'ordinal_risk_optimal',

    # ⑤ 每个 epoch 绘制并保存普通CM + 代价加权CM对比图
    'plot_cost_weighted_confusion_matrix': True,
}
