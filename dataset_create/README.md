# 数据集构建流水线

按以下顺序执行各脚本，从原始 SAR 和冰图数据生成训练用 NetCDF 文件。

## 步骤说明

| 脚本 | 功能 |
|------|------|
| `0_prescreening_icechart.py` | 预筛选 CIS 冰图，找出含有目标 SOD 类别的日期 |
| `2_Sentinel-1_search.py` | 搜索并下载对应日期的 Sentinel-1 SAR 场景 |
| `3_process_sar.py` | 处理 SAR 原始数据（噪声去除、投影、裁剪） |
| `4_auto_make_labels.py` | 将冰图 Shapefile 栅格化，生成 SIC/SOD/FLOE 标签，打包为 NC 文件 |
| `4b_pack_for_inference.py` | 将无标签场景打包为推理用 NC 文件（无 GT） |
| `5_global_zscore.py` | 计算全局 z-score 归一化统计量（均值/标准差） |
| `5_apply_zscore_from_stats_wa.py` | 将预计算的 z-score 统计量应用到训练集 |
| `5b_apply_zscore_for_inference.py` | 将 z-score 归一化应用到推理集 |

## 编号说明

- 步骤 1 已被步骤 0（预筛选）取代，故编号从 0 跳至 2。
- 步骤 4b 和 5b 是各自主步骤的变体，处理无 GT 标签的推理数据。
- 步骤 5 有两个子变体（`5_global_zscore.py` 计算统计量，`5_apply_zscore_from_stats_wa.py` 应用统计量），针对不同数据来源。

## 运行方式

所有脚本应从**项目根目录**运行，例如：

```bash
python dataset_create/4_auto_make_labels.py
```
