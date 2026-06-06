"""
预筛选 CIS 冰图，找出含有 SOD cls3 / cls4 的日期。

逻辑：直接扫描 Shapefile 的 CT/CA/CB/CC/SA/SB/SC 字段，
复现 4_auto_make_labels.py 中完整的动态阈值判定逻辑，
只有真正能产出 cls3/cls4 标签的多边形才会被计入。
不需要下载或处理任何 SAR 图像，速度极快。

与 4_auto_make_labels.py 一致的判定条件（以 cls3/cls4 为例）:
  1. ct_class > 0（非开阔水域）
  2. 按 SIC 权重聚合各 SOD 分量后，主导 SOD 类别的权重比例 ≥ min_ratio
  3. ct_class ≥ min_ct
  动态阈值 (DYNAMIC_THRESHOLDS_SOD):
    cls3 → (min_ct=9, min_ratio=0.9)
    cls4 → (min_ct=9, min_ratio=0.9)

输出：
  - 终端打印汇总
  - 可选输出 CSV（含日期、shp路径、cls3/cls4 多边形数量及面积占比）

    python 0_prescreening_icechart.py --cis-roots "F:\ZJU\11_Ice\dataset_create\Icechart\CIS_cls34" --target both --output-dates "candidate_dates.txt" --output-cis-folders "candidate_cis_folders.txt" --output-csv "prescreening_result.csv"
  
24年1-3月
24年1月         OK
25年一月        OK
26年1-3月       OK


"""

import argparse
import csv
import glob
import os
import re
import sys
from datetime import datetime
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

import src.utils as utils

# 与 4_auto_make_labels.py 中 DYNAMIC_THRESHOLDS_SOD 完全一致
DYNAMIC_THRESHOLDS_SOD = {
    1: (5, 0.5),
    2: (9, 0.8),
    3: (9, 0.9),   # cls3: Thin FYI
    4: (9, 0.9),   # cls4: Medium/Thick FYI
    5: (9, 0.9),
}

NOT_FILLED = utils.ICECHART_NOT_FILLED_VALUE  # -9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-screen CIS ice chart Shapefiles for SOD cls3/cls4 polygons "
                    "(with full dynamic-threshold check matching 4_auto_make_labels.py)."
    )
    parser.add_argument(
        "--cis-roots",
        nargs="+",
        default=[r"F:\ZJU\11_Ice\dataset_create\Icechart\CIS_cls34"],
        help="One or more CIS root directories to scan.",
    )
    parser.add_argument(
        "--target",
        choices=["cls3", "cls4", "both", "any"],
        default="any",
        help=(
            "Filter mode: "
            "'cls3' = only dates with cls3; "
            "'cls4' = only dates with cls4; "
            "'both' = dates containing BOTH cls3 AND cls4 (mixed); "
            "'any'  = dates with cls3 OR cls4 (default)."
        ),
    )
    parser.add_argument(
        "--min-poly",
        type=int,
        default=1,
        help="Minimum number of qualifying polygons required (default: 1).",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Optional CSV output path for the full result table.",
    )
    parser.add_argument(
        "--output-dates",
        default="",
        help=(
            "Optional output path for candidate_dates.txt. "
            "每行一个 YYYY-MM-DD，供后续手动准备 CIS 文件夹时参考。"
        ),
    )
    parser.add_argument(
        "--output-cis-folders",
        default="",
        help=(
            "Optional output path for candidate_cis_folders.txt. "
            "每行一个冰图文件夹的绝对路径，可直接用于组建只含候选日期的 CIS 目录。"
        ),
    )
    return parser.parse_args()


def extract_date_from_folder(folder_name: str) -> Optional[datetime.date]:
    match = re.search(r"(\d{8})", folder_name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError:
        return None


def find_shapefile(folder_path: str) -> Optional[str]:
    # Top-level first, then recursive
    for pattern in [
        os.path.join(folder_path, "*.shp"),
        os.path.join(folder_path, "**", "*.shp"),
    ]:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0].replace("\\", "/")
    return None


def _safe_int_code(value) -> int:
    """将 Shapefile 字段值稳健转换为 int；无法解析时返回 NOT_FILLED(-9)。"""
    if pd.isna(value):
        return NOT_FILLED
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return NOT_FILLED


def _convert_code(code: int, lookup: dict) -> int:
    """将单个 SIGRID-3 编码映射为训练类别；未知编码返回 mask(255)。"""
    return lookup.get(code, lookup["mask"])


def _determine_sod_class(ct: int, ca: int, cb: int, cc: int,
                          sa: int, sb: int, sc: int) -> int:
    """
    复现 4_auto_make_labels.py 中的完整 SOD 判定逻辑：
      1. 将原始 SIGRID-3 编码通过 SIC_LOOKUP / SOD_LOOKUP 转换为训练类别
      2. CT>0 且 CA 为空时令 CA = CT（官方特例）
      3. 按 SIC 权重聚合同类 SOD 分量（sod_weight_by_class）
      4. 取权重最大的候选 SOD 类，验证动态阈值后返回
    返回值：最终 SOD 训练类别，或 SOD_LOOKUP['invalid'](-9) 表示不满足条件。
    """
    # --- Step 1: 编码转换 ---
    ct_class = _convert_code(ct, utils.SIC_LOOKUP)
    ca_class = _convert_code(ca, utils.SIC_LOOKUP) if ca != NOT_FILLED else ca
    cb_class = _convert_code(cb, utils.SIC_LOOKUP) if cb != NOT_FILLED else cb
    cc_class = _convert_code(cc, utils.SIC_LOOKUP) if cc != NOT_FILLED else cc

    sa_class = _convert_code(sa, utils.SOD_LOOKUP) if sa != NOT_FILLED else sa
    sb_class = _convert_code(sb, utils.SOD_LOOKUP) if sb != NOT_FILLED else sb
    sc_class = _convert_code(sc, utils.SOD_LOOKUP) if sc != NOT_FILLED else sc

    # --- Step 2: 官方特例 —— CT 有值但 CA 为空时令 CA = CT class ---
    if ct_class > utils.SIC_LOOKUP[0] and ca_class == NOT_FILLED:
        ca_class = ct_class

    partial_sic = [ca_class, cb_class, cc_class]
    partial_sod = [sa_class, sb_class, sc_class]

    if ct_class <= 0:
        return utils.SOD_LOOKUP["invalid"]

    # --- Step 3: 按 SIC 权重聚合同类 SOD 分量 ---
    sod_weight_by_class: dict = {}
    for sod_cls, sic_w in zip(partial_sod, partial_sic):
        if sod_cls in (NOT_FILLED, utils.SOD_LOOKUP["mask"]):
            continue
        if sic_w in (NOT_FILLED, utils.SIC_LOOKUP["mask"]) or sic_w <= 0:
            continue
        sod_weight_by_class[sod_cls] = sod_weight_by_class.get(sod_cls, 0) + sic_w

    if not sod_weight_by_class:
        return utils.SOD_LOOKUP["invalid"]

    # --- Step 4: 取权重最大的候选类，验证动态阈值 ---
    candidate_cls, candidate_weight = max(
        sod_weight_by_class.items(),
        key=lambda item: (item[1], -item[0])
    )
    sod_ratio = candidate_weight / ct_class  # ct_class > 0 已保证

    min_ct, min_ratio = DYNAMIC_THRESHOLDS_SOD.get(candidate_cls, (9, 0.9))
    if ct_class >= min_ct and sod_ratio >= min_ratio:
        return candidate_cls

    return utils.SOD_LOOKUP["invalid"]


def screen_shapefile(shp_path: str) -> dict:
    """
    读取单个 Shapefile，对每个多边形执行与 4_auto_make_labels.py 完全一致的
    SOD 动态阈值判定，统计通过阈值的 cls3 / cls4 多边形数量和面积。

    Returns a dict with:
      has_cls3, has_cls4,
      n_poly_cls3, n_poly_cls4, n_poly_both,
      area_cls3, area_cls4  (sum of AREA field if available, else polygon count as proxy)
    """
    gdf = gpd.read_file(shp_path)
    gdf.columns = gdf.columns.str.strip().str.upper()

    # 跳过陆地多边形
    if "POLY_TYPE" in gdf.columns:
        gdf = gdf[gdf["POLY_TYPE"].astype(str).str.strip().str.upper() != "L"]

    # 必须有 SA（至少一个 SOD 分量字段）
    if "SA" not in gdf.columns:
        return {
            "has_cls3": False, "has_cls4": False,
            "n_poly_cls3": 0, "n_poly_cls4": 0, "n_poly_both": 0,
            "area_cls3": 0.0, "area_cls4": 0.0,
        }

    has_area = "AREA" in gdf.columns
    n_cls3 = n_cls4 = n_both = 0
    area_cls3 = area_cls4 = 0.0

    for _, row in gdf.iterrows():
        ct = _safe_int_code(row.get("CT", NOT_FILLED))
        ca = _safe_int_code(row.get("CA", NOT_FILLED))
        cb = _safe_int_code(row.get("CB", NOT_FILLED))
        cc = _safe_int_code(row.get("CC", NOT_FILLED))
        sa = _safe_int_code(row.get("SA", NOT_FILLED))
        sb = _safe_int_code(row.get("SB", NOT_FILLED))
        sc = _safe_int_code(row.get("SC", NOT_FILLED))

        sod_cls = _determine_sod_class(ct, ca, cb, cc, sa, sb, sc)

        is_cls3 = (sod_cls == 3)
        is_cls4 = (sod_cls == 4)
        area = float(row["AREA"]) if has_area else 1.0

        if is_cls3:
            n_cls3 += 1
            area_cls3 += area
        if is_cls4:
            n_cls4 += 1
            area_cls4 += area
        if is_cls3 and is_cls4:   # 理论上单多边形只有一个主导类，保留以防未来扩展
            n_both += 1

    return {
        "has_cls3": n_cls3 > 0,
        "has_cls4": n_cls4 > 0,
        "n_poly_cls3": n_cls3,
        "n_poly_cls4": n_cls4,
        "n_poly_both": n_both,
        "area_cls3": area_cls3,
        "area_cls4": area_cls4,
    }


def collect_all_shapefiles(cis_roots: list) -> list:
    """Walk all CIS root dirs, return list of (date, shp_path, root_dir)."""
    entries = []
    for root in cis_roots:
        root = root.replace("\\", "/")
        if not os.path.isdir(root):
            print(f"[警告] 目录不存在，已跳过: {root}")
            continue
        for name in sorted(os.listdir(root)):
            folder_path = os.path.join(root, name)
            if not os.path.isdir(folder_path):
                continue
            date = extract_date_from_folder(name)
            if date is None:
                continue
            shp = find_shapefile(folder_path)
            if shp is None:
                continue
            entries.append((date, shp, root))
    return entries


def passes_filter(stats: dict, target: str, min_poly: int) -> bool:
    if target == "cls3":
        return stats["n_poly_cls3"] >= min_poly
    if target == "cls4":
        return stats["n_poly_cls4"] >= min_poly
    if target == "both":
        return stats["n_poly_cls3"] >= min_poly and stats["n_poly_cls4"] >= min_poly
    # "any"
    return (stats["n_poly_cls3"] + stats["n_poly_cls4"]) >= min_poly


def main() -> None:
    args = parse_args()

    entries = collect_all_shapefiles(args.cis_roots)
    if not entries:
        print("未找到任何 Shapefile。请检查 --cis-roots 路径。")
        return

    print(f"共找到 {len(entries)} 个冰图文件夹，开始扫描...\n")

    results = []
    for idx, (date, shp_path, root_dir) in enumerate(entries, start=1):
        stats = screen_shapefile(shp_path)
        results.append({
            "date": str(date),
            "shp_path": shp_path,
            "cis_root": root_dir,
            **stats,
        })
        if idx % 10 == 0 or idx == len(entries):
            print(f"  已处理 {idx}/{len(entries)}: {date}  "
                  f"cls3多边形={stats['n_poly_cls3']}  cls4多边形={stats['n_poly_cls4']}")

    # Apply filter
    filtered = [r for r in results if passes_filter(r, args.target, args.min_poly)]

    print(f"\n{'='*60}")
    print(f"筛选条件: target={args.target}, min_poly={args.min_poly}")
    print(f"总冰图数: {len(results)}  |  满足条件: {len(filtered)}")
    print(f"{'='*60}")

    if filtered:
        # Classify
        only3 = [r for r in filtered if r["has_cls3"] and not r["has_cls4"]]
        only4 = [r for r in filtered if r["has_cls4"] and not r["has_cls3"]]
        mixed = [r for r in filtered if r["has_cls3"] and r["has_cls4"]]

        print(f"\n  仅含 cls3: {len(only3)} 个")
        print(f"  仅含 cls4: {len(only4)} 个")
        print(f"  cls3+cls4 混合: {len(mixed)} 个")

        print(f"\n{'─'*60}")
        print(f"{'日期':<12} {'类型':<12} {'cls3多边形':>10} {'cls4多边形':>10}  shp路径")
        print(f"{'─'*60}")
        for r in filtered:
            kind = ("mixed" if (r["has_cls3"] and r["has_cls4"])
                    else "cls3-only" if r["has_cls3"]
                    else "cls4-only")
            print(f"{r['date']:<12} {kind:<12} {r['n_poly_cls3']:>10} {r['n_poly_cls4']:>10}  {r['shp_path']}")
    else:
        print("未找到满足条件的冰图日期。")

    if args.output_csv:
        out_csv = args.output_csv.replace("\\", "/")
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "date", "has_cls3", "has_cls4",
                "n_poly_cls3", "n_poly_cls4", "n_poly_both",
                "area_cls3", "area_cls4", "shp_path", "cis_root",
            ])
            writer.writeheader()
            for r in filtered:
                writer.writerow(r)
        print(f"详细统计 CSV 已保存至: {out_csv}")

    # --- 输出候选日期列表（供 2_Sentinel-1_search.py 使用前参考）---
    if args.output_dates:
        out_dates = args.output_dates.replace("\\", "/")
        os.makedirs(os.path.dirname(out_dates) or ".", exist_ok=True)
        with open(out_dates, "w", encoding="utf-8") as f:
            for r in filtered:
                f.write(r["date"] + "\n")
        print(f"候选日期列表已保存至: {out_dates}")
        print("  → 下一步：将 CIS 文件夹中只保留上述日期的子目录，再运行 2_Sentinel-1_search.py")

    # --- 输出候选冰图文件夹路径列表（方便手动复制/移动）---
    if args.output_cis_folders:
        out_folders = args.output_cis_folders.replace("\\", "/")
        os.makedirs(os.path.dirname(out_folders) or ".", exist_ok=True)
        with open(out_folders, "w", encoding="utf-8") as f:
            for r in filtered:
                # shp_path 形如 .../rgc_a09_20240205_CEXPRHB/05022024.shp
                # 取其上一级即冰图文件夹
                folder = os.path.dirname(r["shp_path"])
                f.write(folder + "\n")
        print(f"候选冰图文件夹列表已保存至: {out_folders}")
        print("  → 可将上述文件夹复制/移动到新建的 CIS 目录，再将该目录路径填入 2_Sentinel-1_search.py")


if __name__ == "__main__":
    main()
