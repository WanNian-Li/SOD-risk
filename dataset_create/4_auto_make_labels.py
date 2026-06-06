'''
该脚本实现了从 SAR 图像和对应冰图 Shapefile 自动生成训练用标签矩阵，并打包为 NetCDF 文件的全流程。
主要功能包括：
1. 自动从 SAR 文件夹名提取成像日期，并匹配同日或 ±1 天的冰图 Shapefile。
2. 直接读取 SAR 图像的空间信息，在内存中对齐并栅格化冰图多边形，同时根据 SIGRID-3 编码映射生成 SIC、SOD、FLOE 标签矩阵。
3. 处理 SAR 无效像素（如雷达黑边）和冰图陆地区，确保标签矩阵与 SAR 输入完全对齐。


最终SAR黑边和陆地区域的输入变量被替换为 -9999.0（nodata）：
- nersc_sar_primary (HH)
- nersc_sar_secondary (HV)
- sar_incidenceangle
- 3个GLCM纹理特征

黑边和陆地区域的标签值均为 255（mask）：
- SIC
- SOD
- FLOE

'''
import os
import re
import glob
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
import xarray as xr
import src.utils as utils
import time
from datetime import datetime, timedelta
from functools import wraps
import h5py

SAR_ROOT_DIR = r"F:\ZJU\11_Ice\dataset_create\S1_process_CAA"
ICECHART_ROOT_DIR = r"F:\ZJU\11_Ice\dataset_create\Icechart\CIS_CAA"
OUTPUT_NC_DIR = r"F:\ZJU\11_Ice\dataset_create\data_nc_CAA"
SKIP_EXISTING_NC = True  # True: 已存在 NC 自动跳过（断点续跑）；修复标注后需设为 False 以强制重生成

# 断续重传配置
ENABLE_RESUME_STATE = True
RESUME_STATE_FILENAME = "auto_make_labels_resume_state.json"
MAX_RETRY_PER_SCENE = 2      # 每个场景失败后最多重试次数（总尝试 = 1 + MAX_RETRY_PER_SCENE）
RETRY_WAIT_SECONDS = 3       # 单次失败后的等待秒数

HH_FILENAME_CANDIDATES = [
    "Sigma0_HH_corrected.img",
    "Sigma0_HH.img",
]
HV_FILENAME_CANDIDATES = [
    "Sigma0_HV_pass.img",
    "Sigma0_HV.img",
]
INC_FILENAME_CANDIDATES = [
    "incidenceAngleFromEllipsoid.img",
    "localIncidenceAngle.img",
]

DYNAMIC_THRESHOLDS_SOD = {
    # 标签1 (NI 新冰)：总浓度下限设为 50% (CT>=5)，主导比例下限 50% (0.5)
    1: (5, 0.5),   
    # 标签2 (GI/GWI)：总浓度下限设为 90% (CT>=9)，主导比例下限 80% (0.8)
    2: (9, 0.8),   
    # 标签3, 4, 5 (ThinFI, MedFI, ThickFI)：总浓度下限 90% (CT>=9)，主导比例下限 90% (0.9)
    3: (9, 0.9),   
    4: (9, 0.9),   
    5: (9, 0.9),   
}

# FLOE 同样适用此逻辑，这里我给出一组推荐的高纯度默认值，你可以根据实际图表分布随时修改
DYNAMIC_THRESHOLDS_FLOE = {
    1: (9, 0.8),   # Cake Ice (通常较碎，可稍微放宽比例到0.8)
    2: (9, 0.9),   # Small floe
    3: (9, 0.9),   # Medium floe
    4: (9, 0.9),   # Big floe
    5: (9, 0.9),   # Vast floe
    6: (9, 0.9),   # Bergs
}

def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"该脚本函数 {func.__name__} 耗时: {end - start:.6f} 秒")
        return result
    return wrapper


def extract_sar_date_from_name(folder_name):
    """
    从 SAR 文件夹名中提取成像日期（YYYYMMDD）。
    例如：S1A_..._20250224T122232_..._glcm.data -> 2025-02-24
    """
    match = re.search(r"_(\d{8})T\d{6}_", folder_name)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d").date()


def extract_chart_date_from_name(folder_name):
    """
    从冰图文件夹名中提取日期（YYYYMMDD）。
    例如：rgc_a09_20250224_CEXPRHB -> 2025-02-24
    """
    match = re.search(r"(\d{8})", folder_name)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d").date()


def find_shapefile_in_chart_folder(chart_folder):
    """在冰图文件夹中查找 .shp 文件（优先顶层，其次递归）。"""
    top_level = glob.glob(os.path.join(chart_folder, "*.shp"))
    if top_level:
        return top_level[0].replace('\\', '/')

    recursive = glob.glob(os.path.join(chart_folder, "**", "*.shp"), recursive=True)
    if recursive:
        return recursive[0].replace('\\', '/')

    return None


def index_icecharts_by_date(icechart_root_dir):
    """建立冰图索引：{日期: [shp_path1, shp_path2, ...]}。"""
    chart_index = {}
    chart_dirs = sorted(glob.glob(os.path.join(icechart_root_dir, "*")))

    for chart_dir in chart_dirs:
        if not os.path.isdir(chart_dir):
            continue

        chart_date = extract_chart_date_from_name(os.path.basename(chart_dir))
        if chart_date is None:
            continue

        shp_path = find_shapefile_in_chart_folder(chart_dir)
        if shp_path is None:
            continue

        chart_index.setdefault(chart_date, []).append(shp_path)

    return chart_index


def select_best_chart(sar_date, chart_index):
    """
    选择最匹配冰图：
    1) 同日；2) 前1天/后1天（按绝对时间差最小）。
    """
    candidate_dates = [sar_date, sar_date - timedelta(days=1), sar_date + timedelta(days=1)]
    available = [d for d in candidate_dates if d in chart_index and len(chart_index[d]) > 0]
    if not available:
        return None, None

    best_date = sorted(available, key=lambda d: (abs((d - sar_date).days), d))[0]
    return best_date, chart_index[best_date][0]


def find_band_file(sar_data_dir, filename_candidates):
    """在 SAR 文件夹中按候选文件名查找波段文件。"""
    for filename in filename_candidates:
        full_path = os.path.join(sar_data_dir, filename)
        if os.path.exists(full_path):
            return full_path.replace('\\', '/')
    return None


def find_glcm_files(sar_data_dir):
    """
    自动发现 GLCM 波段文件。
    约定匹配：Sigma0_HH_corrected_*.img（排除基础波段 Sigma0_HH_corrected.img）。
    """
    pattern = os.path.join(sar_data_dir, "Sigma0_HH_*.img")
    glcm_paths = sorted(glob.glob(pattern))
    return [p.replace('\\', '/') for p in glcm_paths]


def _safe_int_code(value, default=utils.ICECHART_NOT_FILLED_VALUE):
    """将 shapefile 字段值稳健转换为 int；无法解析时返回 default。"""
    if pd.isna(value):
        return default
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _convert_code(code, lookup):
    """将单个 SIGRID3 编码映射为训练类别；未知编码返回 mask。"""
    return lookup.get(code, lookup['mask'])


def _now_iso():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_resume_state_path(output_nc_dir):
    return os.path.join(output_nc_dir, RESUME_STATE_FILENAME).replace('\\', '/')


def load_resume_state(state_path):
    """读取断点状态文件，不存在时返回默认结构。"""
    default_state = {
        "meta": {
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "state_version": 1,
        },
        "scenes": {}
    }

    if not ENABLE_RESUME_STATE:
        return default_state

    if not os.path.exists(state_path):
        return default_state

    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        if not isinstance(state, dict):
            return default_state
        if "meta" not in state or not isinstance(state["meta"], dict):
            state["meta"] = default_state["meta"]
        if "scenes" not in state or not isinstance(state["scenes"], dict):
            state["scenes"] = {}
        return state
    except Exception as e:
        print(f"[断续重传] 状态文件读取失败，将使用空状态继续：{e}")
        return default_state


def save_resume_state(state_path, state):
    """原子写入状态文件，避免中断导致 JSON 损坏。"""
    if not ENABLE_RESUME_STATE:
        return

    state["meta"]["updated_at"] = _now_iso()
    tmp_path = f"{state_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, state_path)


def update_scene_state(state, scene_name, **kwargs):
    """更新单场景状态并返回更新后的记录。"""
    scene_info = state["scenes"].get(scene_name, {})
    scene_info.update(kwargs)
    scene_info["updated_at"] = _now_iso()
    state["scenes"][scene_name] = scene_info
    return scene_info


def get_scene_attempts(state, scene_name):
    scene_info = state["scenes"].get(scene_name, {})
    attempts = scene_info.get("attempts", 0)
    if isinstance(attempts, int) and attempts >= 0:
        return attempts
    return 0

def rasterize_and_map_labels(shp_path, ref_img_path):
    """
    直接读取 SAR 空间信息，将 Shapefile 在内存中对齐并栅格化，同时完成属性映射。
    """
    print("\n[1/4] 正在提取 SAR 图像尺寸和所用坐标系...")
    with rasterio.open(ref_img_path) as src:
        sar_shape = (src.height, src.width)
        sar_transform = src.transform
        sar_crs = src.crs
    # print(f"      - SAR 尺寸: {sar_shape}")
    # print(f"      - SAR 坐标系: {sar_crs}")

    print("\n[2/4] 正在加载并重投影矢量冰图...")
    gdf = gpd.read_file(shp_path)
    
    # --- 新增代码：打印原始表头信息 ---
    # print(f"      - 原始 GeoDataFrame 表头/属性字段为: \n        {list(gdf.columns)}")
    
    # 自动将冰图多边形的坐标系转换成与 SAR 图像一模一样的坐标系，确保后续栅格化时空间对齐正确
    gdf = gdf.to_crs(sar_crs) 
    
    # 重置索引，防止由于原始 Shapefile 索引不唯一导致的 "cannot reindex on an axis with duplicate labels" 错误
    gdf = gdf.reset_index(drop=True)

    # 自动生成 POLY_ID (替代 QGIS 里的 $id)
    gdf['POLY_ID'] = range(1, len(gdf) + 1)

    # 将列名统一转大写，防止因为字段名大小写报错
    gdf.columns = gdf.columns.str.strip().str.upper()
    if not gdf.columns.is_unique:
        print("发现重复字段，已保留首个:", gdf.columns[gdf.columns.duplicated()].tolist())
        gdf = gdf.loc[:, ~gdf.columns.duplicated()]
    if 'GEOMETRY' in gdf.columns:
        gdf = gdf.set_geometry('GEOMETRY')

    print("\n[3/4] 正在内存中执行矢量栅格化烧录 (Rasterize)...")
    # 构建生成器：(多边形几何体, 烧录的数值)
    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf['POLY_ID']))
    
    # 生成与 SAR 矩阵完全对齐的 ID 矩阵 (背景 0 代表无数据/陆地)
    poly_id_matrix = features.rasterize(
        shapes=shapes,
        out_shape=sar_shape,
        transform=sar_transform,
        fill=0, 
        dtype='uint32'
    )

    # 额外构建陆地掩膜：仅标记 POLY_TYPE == 'L' 的多边形像素。
    land_mask = np.zeros(sar_shape, dtype=bool)
    if 'POLY_TYPE' in gdf.columns:
        poly_type_series = gdf['POLY_TYPE'].astype(str).str.strip().str.lower()
        land_polys = gdf.loc[poly_type_series == 'l']
        if len(land_polys) > 0:
            land_shapes = ((geom, 1) for geom in land_polys.geometry)
            land_mask = features.rasterize(
                shapes=land_shapes,
                out_shape=sar_shape,
                transform=sar_transform,
                fill=0,
                dtype='uint8'
            ).astype(bool)

    print("\n[4/4] 正在执行 SIC, SOD, FLOE 物理属性映射...")
    # 先使用有符号整型承载 invalid=-9，避免写入 uint8 后变成 247。
    sic_matrix = np.full(sar_shape, utils.SIC_LOOKUP['mask'], dtype=np.int16)
    sod_matrix = np.full(sar_shape, utils.SOD_LOOKUP['mask'], dtype=np.int16)
    floe_matrix = np.full(sar_shape, utils.FLOE_LOOKUP['mask'], dtype=np.int16)

    # 遍历所有有效的冰区多边形，进行数值填充
    # (复刻原本在 qgis_to_ready_labels.py 中的算法逻辑)
    unique_ids = np.unique(poly_id_matrix)
    # 这是一个布尔掩码，索引与 gdf 相同
    valid_poly_mask = gdf['POLY_ID'].isin(unique_ids)
    
    # 如果索引有重复，强行重置一次并在此时过滤
    if not gdf.index.is_unique:
        print("警告：GeoDataFrame 索引仍有重复，正在尝试强制重置索引...")
        gdf = gdf.reset_index(drop=True)
        valid_poly_mask = gdf['POLY_ID'].isin(unique_ids)

    valid_polys = gdf[valid_poly_mask]
    
    for _, row in valid_polys.iterrows():
        poly_id = row['POLY_ID']
        
        # 提取 SIGRID-3 参数，处理缺失值
        ct = _safe_int_code(row['CT'])
        ca = _safe_int_code(row['CA'])
        cb = _safe_int_code(row['CB'])
        cc = _safe_int_code(row['CC'])
        sa = _safe_int_code(row['SA'])
        sb = _safe_int_code(row['SB'])
        sc = _safe_int_code(row['SC'])
        fa = _safe_int_code(row['FA'])
        fb = _safe_int_code(row['FB'])
        fc = _safe_int_code(row['FC'])
        poly_type = str(row['POLY_TYPE']).strip().lower() if 'POLY_TYPE' in row.index else ''

        # 1) 先把 SIC/SOD/FLOE 各字段编码映射到训练类别（与官方流程一致）
        ct_class = _convert_code(ct, utils.SIC_LOOKUP)
        ca_class = _convert_code(ca, utils.SIC_LOOKUP) if ca != utils.ICECHART_NOT_FILLED_VALUE else ca
        cb_class = _convert_code(cb, utils.SIC_LOOKUP) if cb != utils.ICECHART_NOT_FILLED_VALUE else cb
        cc_class = _convert_code(cc, utils.SIC_LOOKUP) if cc != utils.ICECHART_NOT_FILLED_VALUE else cc

        sa_class = _convert_code(sa, utils.SOD_LOOKUP) if sa != utils.ICECHART_NOT_FILLED_VALUE else sa
        sb_class = _convert_code(sb, utils.SOD_LOOKUP) if sb != utils.ICECHART_NOT_FILLED_VALUE else sb
        sc_class = _convert_code(sc, utils.SOD_LOOKUP) if sc != utils.ICECHART_NOT_FILLED_VALUE else sc

        fa_class = _convert_code(fa, utils.FLOE_LOOKUP) if fa != utils.ICECHART_NOT_FILLED_VALUE else fa
        fb_class = _convert_code(fb, utils.FLOE_LOOKUP) if fb != utils.ICECHART_NOT_FILLED_VALUE else fb
        fc_class = _convert_code(fc, utils.FLOE_LOOKUP) if fc != utils.ICECHART_NOT_FILLED_VALUE else fc

        # 2) 官方特例：CT 有值但 CA 为空时，令 CA=CT（按“已转换后的 SIC 类”补齐）
        if ct_class > utils.SIC_LOOKUP[0] and ca_class == utils.ICECHART_NOT_FILLED_VALUE:
            ca_class = ct_class

        partial_sic = np.array([ca_class, cb_class, cc_class], dtype=np.int32)
        partial_sod = np.array([sa_class, sb_class, sc_class], dtype=np.int32)
        partial_floe = np.array([fa_class, fb_class, fc_class], dtype=np.int32)

        # --- SIC (密集度) ---
        sic_class = ct_class
        if poly_type == 'w':
            sic_class = utils.SIC_LOOKUP[0]

        # 3) SOD 改为按类别聚合部分浓度，再判断 dominant；FLOE 保持原实现
        floe_partial_add = np.zeros(3, dtype=np.int32)
        compare_indexes = [(0, 1), (0, 2), (1, 2)]

        sod_weight_by_class = {}
        for idx in range(3):
            sod_cls = int(partial_sod[idx])
            sic_w = int(partial_sic[idx])
            if sod_cls in (utils.ICECHART_NOT_FILLED_VALUE, utils.SOD_LOOKUP['mask']):
                continue
            if sic_w in (utils.ICECHART_NOT_FILLED_VALUE, utils.SIC_LOOKUP['mask']) or sic_w <= 0:
                continue
            sod_weight_by_class[sod_cls] = sod_weight_by_class.get(sod_cls, 0) + sic_w

        for a, b in compare_indexes:
            if partial_floe[a] != utils.ICECHART_NOT_FILLED_VALUE and partial_floe[a] == partial_floe[b]:
                floe_partial_add[a] += partial_sic[b]

        tmp_floe_added = partial_sic + floe_partial_add

        sod_class = utils.SOD_LOOKUP['invalid']
        floe_class = utils.FLOE_LOOKUP['invalid']

        if ct_class > 0:
            # ---------------- SOD 动态阈值判定 ----------------
            if len(sod_weight_by_class) > 0:
                candidate_sod_class, candidate_sod_weight = max(
                    sod_weight_by_class.items(),
                    key=lambda item: (item[1], -item[0])
                )
                sod_ratio = np.divide(candidate_sod_weight, ct_class)

                if np.isfinite(sod_ratio):
                    # 从顶部配置字典获取该类的专属阈值；若类别不存在，采用默认兜底 (CT>=9, ratio>=0.9)
                    min_ct, min_ratio = DYNAMIC_THRESHOLDS_SOD.get(candidate_sod_class, (9, 0.9))
                    # 同时校验 总浓度 和 比例
                    if ct_class >= min_ct and sod_ratio >= min_ratio:
                        sod_class = candidate_sod_class

            # ---------------- FLOE 动态阈值判定 ----------------
            candidate_floe_idx = int(np.argmax(tmp_floe_added))
            candidate_floe_class = int(partial_floe[candidate_floe_idx])
            floe_ratio = np.divide(tmp_floe_added[candidate_floe_idx], ct_class)

            if np.isfinite(floe_ratio):
                min_ct, min_ratio = DYNAMIC_THRESHOLDS_FLOE.get(candidate_floe_class, (9, 0.9))
                if ct_class >= min_ct and floe_ratio >= min_ratio:
                    floe_class = candidate_floe_class

            # 保留原有的 fastice 绝对优先级特判逻辑
            if np.any(partial_floe == utils.FLOE_LOOKUP['fastice_class']):
                floe_class = utils.FLOE_LOOKUP['fastice_class']

        # 找到矩阵中属于该多边形的像素，直接赋上计算好的类别值
        poly_pixel_mask = (poly_id_matrix == poly_id)
        sic_matrix[poly_pixel_mask] = sic_class
        sod_matrix[poly_pixel_mask] = sod_class
        floe_matrix[poly_pixel_mask] = floe_class

    # 与官方一致：将 ambiguous/not-filled 置为 mask；并保证水域在三个图层一致。
    sod_matrix[sod_matrix == utils.SOD_LOOKUP['invalid']] = utils.SOD_LOOKUP['mask']
    floe_matrix[floe_matrix == utils.FLOE_LOOKUP['invalid']] = utils.FLOE_LOOKUP['mask']

    sod_matrix[sic_matrix == utils.SIC_LOOKUP[0]] = utils.SOD_LOOKUP['water']
    floe_matrix[sic_matrix == utils.SIC_LOOKUP[0]] = utils.FLOE_LOOKUP['water']

    # 移除 unknown 值（99）
    sic_matrix[sic_matrix == utils.ICECHART_UNKNOWN] = utils.SIC_LOOKUP['mask']
    sod_matrix[sod_matrix == utils.ICECHART_UNKNOWN] = utils.SOD_LOOKUP['mask']
    floe_matrix[floe_matrix == utils.ICECHART_UNKNOWN] = utils.FLOE_LOOKUP['mask']

    # 后处理完成后再转回 uint8，保持输出与数据集规范一致。
    return sic_matrix.astype(np.uint8), sod_matrix.astype(np.uint8), floe_matrix.astype(np.uint8), land_mask


def package_to_netcdf(hh_array, hv_array, inc_angle_array, sic_array, sod_array, floe_array, glcm_data_dict, output_nc_path):
    print("\n[打包阶段] 正在构建 xarray.Dataset 并保存...")
    dims = ["y", "x"]
    data_vars = {
        "nersc_sar_primary": (dims, hh_array.astype(np.float32), {"description": "Sigma0 in dB", "polarisation": "HH"}),
        "nersc_sar_secondary": (dims, hv_array.astype(np.float32), {"description": "Sigma0 in dB", "polarisation": "HV"}),
        "sar_incidenceangle": (dims, inc_angle_array.astype(np.float32), {"description": "Incidence angle", "units": "degrees"}),
        "SIC": (dims, sic_array.astype(np.uint8), {"description": "Sea Ice Concentration", "chart_fill_value": 255}),
        "SOD": (dims, sod_array.astype(np.uint8), {"description": "Stage of Development", "chart_fill_value": 255}),
        "FLOE": (dims, floe_array.astype(np.uint8), {"description": "Floe Size", "chart_fill_value": 255}),
    }

    for var_name, var_array in glcm_data_dict.items():
        data_vars[var_name] = (
            dims,
            var_array.astype(np.float32),
            {"description": "GLCM texture feature from Sigma0_HH_corrected"}
        )

    ds = xr.Dataset(
        data_vars=data_vars,
        attrs={"description": "Training data file"}
    )
    # 启用zlib压缩，缩小nc文件体积
    chunk_size = (512, 512)
    compression = dict(zlib=True, complevel=4, shuffle=True, chunksizes=chunk_size)
    encoding_dict = {
        var: compression 
        for var in ds.data_vars
    }

    ds.to_netcdf(output_nc_path, mode='w', format='NETCDF4', engine='h5netcdf', encoding=encoding_dict)
    print(f"NC 文件制作完成，已启用 zlib 压缩: {output_nc_path}")


def build_sar_invalid_mask(hh_array, hv_array, inc_angle_array, hh_nodata, hv_nodata, inc_nodata):
    """构建 SAR 无效像素掩膜：优先 nodata 元数据，兼容 -9999 兜底。"""
    invalid_mask = (~np.isfinite(hh_array)) | (~np.isfinite(hv_array)) | (~np.isfinite(inc_angle_array))

    if hh_nodata is not None and np.isfinite(hh_nodata):
        invalid_mask |= (hh_array == hh_nodata)
    if hv_nodata is not None and np.isfinite(hv_nodata):
        invalid_mask |= (hv_array == hv_nodata)
    if inc_nodata is not None and np.isfinite(inc_nodata):
        invalid_mask |= (inc_angle_array == inc_nodata)

    # 兼容历史流程中常见的统一无效填充值。
    invalid_mask |= (hh_array <= -9990.0) | (hv_array <= -9990.0) | (inc_angle_array <= -9990.0)
    return invalid_mask



def process_single_scene(sar_dir, shp_path, output_nc_path):
    hh_img_path = find_band_file(sar_dir, HH_FILENAME_CANDIDATES)
    hv_img_path = find_band_file(sar_dir, HV_FILENAME_CANDIDATES)
    inc_angle_path = find_band_file(sar_dir, INC_FILENAME_CANDIDATES)

    if hh_img_path is None or hv_img_path is None or inc_angle_path is None:
        missing = []
        if hh_img_path is None:
            missing.append("HH")
        if hv_img_path is None:
            missing.append("HV")
        if inc_angle_path is None:
            missing.append("incidence angle")
        raise FileNotFoundError(f"场景缺少必要波段文件: {', '.join(missing)}")

    # 1. 栅格化与映射
    sic, sod, floe, land_mask = rasterize_and_map_labels(shp_path, hh_img_path)

    # 2. 读取 SAR 数据
    print("\n[读取阶段] 正在从 SNAP .data 文件夹读取 SAR 波段矩阵...")
    with rasterio.open(hh_img_path) as src:
        hh_array = src.read(1)
        hh_nodata = src.nodata
    with rasterio.open(hv_img_path) as src:
        hv_array = src.read(1)
        hv_nodata = src.nodata
    with rasterio.open(inc_angle_path) as src:
        inc_angle_array = src.read(1)
        inc_nodata = src.nodata

    # 2.1 读取 GLCM 波段（自动发现）
    glcm_data_dict = {}
    glcm_paths = find_glcm_files(sar_dir)
    if len(glcm_paths) > 0:
        print(f"[读取阶段] 检测到 GLCM 波段数量: {len(glcm_paths)}")
        for glcm_path in glcm_paths:
            glcm_name = os.path.splitext(os.path.basename(glcm_path))[0]
            var_name = f"glcm_{glcm_name.replace('Sigma0_HH_corrected_', '').lower()}"
            with rasterio.open(glcm_path) as src:
                glcm_data_dict[var_name] = src.read(1)
    else:
        print("[读取阶段] 未检测到 GLCM 波段，将仅打包 HH/HV/角度与标签。")

    # 雷达黑边/无效值掩膜处理（优先使用 nodata 元数据，兼容 -9999）
    invalid_mask = build_sar_invalid_mask(
        hh_array, hv_array, inc_angle_array,
        hh_nodata, hv_nodata, inc_nodata,
    )

    # 最终掩膜 = SAR 无效区 + 冰图陆地区。
    final_mask = invalid_mask | land_mask

    sic[final_mask] = 255
    sod[final_mask] = 255
    floe[final_mask] = 255

    bad_sic = int(np.sum(final_mask & (sic != 255)))
    bad_sod = int(np.sum(final_mask & (sod != 255)))
    bad_floe = int(np.sum(final_mask & (floe != 255)))
    invalid_count = int(np.sum(invalid_mask))
    land_count = int(np.sum(land_mask))
    final_count = int(np.sum(final_mask))
    total_count = int(final_mask.size)
    print(f"[掩膜检查] SAR 无效像素: {invalid_count}/{total_count} ({invalid_count / max(1, total_count):.2%})")
    print(f"[掩膜检查] 陆地像素(POLY_TYPE='L'): {land_count}/{total_count} ({land_count / max(1, total_count):.2%})")
    print(f"[掩膜检查] 最终掩膜像素: {final_count}/{total_count} ({final_count / max(1, total_count):.2%})")
    print(f"[掩膜检查] final_mask 区域中非 255 标签数量: SIC={bad_sic}, SOD={bad_sod}, FLOE={bad_floe}")
    if bad_sic > 0 or bad_sod > 0 or bad_floe > 0:
        raise ValueError("final_mask 区域内存在非 255 标签，请检查掩膜规则与标签赋值流程。")

    hh_array[final_mask] = -9999.0
    hv_array[final_mask] = -9999.0
    inc_angle_array[final_mask] = -9999.0

    for var_name, glcm_array in glcm_data_dict.items():
        if glcm_array.shape != final_mask.shape:
            raise ValueError(f"GLCM 变量尺寸与掩膜不一致: {var_name}, {glcm_array.shape} vs {final_mask.shape}")
        glcm_array[final_mask] = -9999.0

    # 3. 打包为 NetCDF（含可用 GLCM）
    package_to_netcdf(hh_array, hv_array, inc_angle_array, sic, sod, floe, glcm_data_dict, output_nc_path)


@timer_decorator
def main():
    print("====== 纯 Python 全自动标注批处理流水线启动 ======")
    os.makedirs(OUTPUT_NC_DIR, exist_ok=True)

    state_path = get_resume_state_path(OUTPUT_NC_DIR)
    resume_state = load_resume_state(state_path)
    if ENABLE_RESUME_STATE:
        print(f"[断续重传] 状态文件: {state_path}")
        print(f"[断续重传] 已记录场景数: {len(resume_state['scenes'])}")

    sar_dirs = sorted(glob.glob(os.path.join(SAR_ROOT_DIR, "*_glcm.data")))
    if len(sar_dirs) == 0:
        print(f"未在目录中找到 SAR 场景: {SAR_ROOT_DIR}")
        return

    chart_index = index_icecharts_by_date(ICECHART_ROOT_DIR)
    if len(chart_index) == 0:
        print(f"未在目录中找到可用冰图: {ICECHART_ROOT_DIR}")
        return

    print(f"检测到 SAR 场景数量: {len(sar_dirs)}")
    print(f"检测到可用冰图日期数量: {len(chart_index)}")

    success_count = 0
    skip_count = 0
    skipped_existing_count = 0
    resumed_success_skip_count = 0
    exhausted_retry_count = 0

    for idx, sar_dir in enumerate(sar_dirs, start=1):
        sar_name = os.path.basename(sar_dir)
        print("\n" + "=" * 88)
        print(f"[{idx}/{len(sar_dirs)}] 正在处理: {sar_name}")

        scene_prefix = sar_name.replace("_glcm.data", "")
        output_nc_path = os.path.join(OUTPUT_NC_DIR, f"{scene_prefix}.nc").replace('\\', '/')

        scene_state = resume_state["scenes"].get(sar_name, {})
        if ENABLE_RESUME_STATE and scene_state.get("status") == "success" and os.path.exists(output_nc_path):
            resumed_success_skip_count += 1
            print(f"  -> 断续重传跳过：状态文件记录该场景已成功且输出存在: {output_nc_path}")
            continue

        sar_date = extract_sar_date_from_name(sar_name)
        if sar_date is None:
            print("  -> 跳过：无法从 SAR 文件夹名解析日期")
            skip_count += 1
            update_scene_state(
                resume_state,
                sar_name,
                status="skipped_bad_name",
                attempts=get_scene_attempts(resume_state, sar_name),
                output_nc_path=output_nc_path,
                last_error="无法从 SAR 文件夹名解析日期"
            )
            save_resume_state(state_path, resume_state)
            continue

        matched_date, matched_shp = select_best_chart(sar_date, chart_index)
        if matched_shp is None:
            print(f"  -> 跳过：未找到匹配冰图（允许同日或 ±1 天），SAR 日期 = {sar_date}")
            skip_count += 1
            update_scene_state(
                resume_state,
                sar_name,
                status="skipped_no_chart",
                attempts=get_scene_attempts(resume_state, sar_name),
                output_nc_path=output_nc_path,
                sar_date=str(sar_date),
                last_error="未找到匹配冰图（允许同日或 ±1 天）"
            )
            save_resume_state(state_path, resume_state)
            continue

        date_delta = (matched_date - sar_date).days
        print(f"  -> SAR 日期: {sar_date}, 匹配冰图日期: {matched_date}, 天数差: {date_delta:+d}")
        print(f"  -> 冰图 Shapefile: {matched_shp}")

        if SKIP_EXISTING_NC and os.path.exists(output_nc_path):
            skipped_existing_count += 1
            print(f"  -> 跳过：目标 NC 已存在（断点续跑）: {output_nc_path}")
            update_scene_state(
                resume_state,
                sar_name,
                status="skipped_existing",
                attempts=get_scene_attempts(resume_state, sar_name),
                output_nc_path=output_nc_path,
                sar_date=str(sar_date),
                matched_date=str(matched_date),
                matched_shp=matched_shp,
                last_error=""
            )
            save_resume_state(state_path, resume_state)
            continue

        max_total_attempts = 1 + MAX_RETRY_PER_SCENE
        prev_attempts = get_scene_attempts(resume_state, sar_name)
        if prev_attempts >= max_total_attempts:
            exhausted_retry_count += 1
            skip_count += 1
            print(f"  -> 跳过：该场景历史失败次数已达上限 {prev_attempts}/{max_total_attempts}，请人工检查后再重跑。")
            continue

        success = False
        for attempt_no in range(prev_attempts + 1, max_total_attempts + 1):
            print(f"  -> 尝试 {attempt_no}/{max_total_attempts}")
            update_scene_state(
                resume_state,
                sar_name,
                status="running",
                attempts=attempt_no,
                output_nc_path=output_nc_path,
                sar_date=str(sar_date),
                matched_date=str(matched_date),
                matched_shp=matched_shp,
                last_error=""
            )
            save_resume_state(state_path, resume_state)

            try:
                process_single_scene(sar_dir, matched_shp, output_nc_path)
                success_count += 1
                success = True
                print(f"  -> 完成输出: {output_nc_path}")
                update_scene_state(
                    resume_state,
                    sar_name,
                    status="success",
                    attempts=attempt_no,
                    output_nc_path=output_nc_path,
                    sar_date=str(sar_date),
                    matched_date=str(matched_date),
                    matched_shp=matched_shp,
                    last_error=""
                )
                save_resume_state(state_path, resume_state)
                break
            except Exception as e:
                err_msg = str(e)
                print(f"  -> 本次处理失败: {err_msg}")
                update_scene_state(
                    resume_state,
                    sar_name,
                    status="failed",
                    attempts=attempt_no,
                    output_nc_path=output_nc_path,
                    sar_date=str(sar_date),
                    matched_date=str(matched_date),
                    matched_shp=matched_shp,
                    last_error=err_msg
                )
                save_resume_state(state_path, resume_state)

                if attempt_no < max_total_attempts:
                    print(f"  -> {RETRY_WAIT_SECONDS} 秒后自动重试...")
                    time.sleep(RETRY_WAIT_SECONDS)

        if not success:
            skip_count += 1
            print(f"  -> 处理失败：已达到最大尝试次数 {max_total_attempts}，该场景本轮跳过。")
    
    print("\n" + "=" * 88)
    print(
        f"批处理完成：成功 {success_count} 个，"
        f"已存在跳过 {skipped_existing_count} 个，"
        f"断续重传跳过 {resumed_success_skip_count} 个，"
        f"失败达上限跳过 {exhausted_retry_count} 个，"
        f"其他跳过/失败 {skip_count} 个，总计 {len(sar_dirs)} 个"
    )

    '''
    最终需要进行归一化，下面代码执行的是局部归一化（单张SAR），但是最终需要在整个训练集上进行全局归一化
    
    # ==========================================
    print("\n[处理阶段] 正在执行 Z-score 归一化...")
    
    def apply_z_score(array, mask):
        # 仅使用有效数据计算统计量
        valid_data = array[~mask]
        mean_val = np.mean(valid_data)
        std_val = np.std(valid_data)
        if std_val == 0:
            print(f"⚠️  标准差为 0，无法归一化！")
            return array # 如果标准差为 0，直接返回原数组
        
        # 归一化
        norm_array = np.copy(array)
        norm_array[~mask] = (valid_data - mean_val) / std_val
        # 将无效区域的特征值设为 NaN，便于网络识别
        norm_array[mask] = np.nan

        mean_val = np.nanmin(norm_array[~mask])  # 计算有效区域的均值
        std_val = np.nanmax(norm_array[~mask])    # 计算有效区域的标准差

        print(f"  -> 最小值: {np.nanmin(norm_array[~mask]):.2f}, 最大值: {np.nanmax(norm_array[~mask]):.2f}")
        return norm_array

    print("- 归一化 HH 波段:")
    hh_array = apply_z_score(hh_array, invalid_mask)
    print("- 归一化 HV 波段:")
    hv_array = apply_z_score(hv_array, invalid_mask)
    print("- 归一化 入射角:")
    inc_angle_array = apply_z_score(inc_angle_array, invalid_mask)
    '''
    

if __name__ == "__main__":
    main()