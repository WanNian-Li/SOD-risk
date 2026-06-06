'''
该脚本用于"实际应用推理"场景：无冰图，仅依赖 SAR + 海岸线矢量库生成可推理 NC。

与 4_auto_make_labels.py 的区别：
- 不读取冰图 Shapefile，也不生成 SIC/SOD/FLOE 标签
- 陆地掩膜来源改为海岸线矢量库（默认 GSHHG full resolution）
- 额外写入 coastline_land_mask 变量，以及 crs_wkt / geo_transform 全局属性
- 最终掩膜（雷达黑边 ∪ 陆地）像素的所有特征置为 -9999.0，
  与训练集 4_auto_make_labels.py 的无效值语义完全一致，
  使下游 5_apply_zscore_from_stats_wa.py 能自然地将陆地纳入 global_valid_mask。

输出 NC 结构：
- nersc_sar_primary         (HH, float32, -9999 for invalid)
- nersc_sar_secondary       (HV, float32, -9999 for invalid)
- sar_incidenceangle        (float32, -9999 for invalid)
- glcm_*                    (float32, -9999 for invalid) [按 SNAP 输出自动发现]
- coastline_land_mask       (uint8, 1=land, 0=sea)
- attrs.crs_wkt             (SAR 坐标系 WKT 字符串)
- attrs.geo_transform       (6 元 float tuple: a, b, c, d, e, f)

下游推理时：
- 5_apply_zscore_from_stats_wa.py 会在其内部重建 global_valid_mask（基于 -9999），
  陆地像素自动成为 invalid，模型前向时被跳过。
- 可视化时可直接读 coastline_land_mask 把陆地单独涂灰，与雷达黑边区分。
'''

import os
import sys
import glob
import json
import time
import argparse
from datetime import datetime
from functools import wraps

import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from shapely.geometry import box
import xarray as xr

# ============================================================
# 配置区
# ============================================================

# 默认路径（可通过 CLI 覆盖）
SAR_ROOT_DIR_DEFAULT = r"F:/ZJU/11_Ice/dataset_create/data_apply/S1_process"
OUTPUT_NC_DIR_DEFAULT = r"F:/ZJU/11_Ice/dataset_create/data_apply/data_nc_inference"

# GSHHG full-resolution 陆地多边形（L1 = 海陆边界）
# 下载：https://www.soest.hawaii.edu/pwessel/gshhg/
# 解压后路径类似 .../GSHHS_shp/f/GSHHS_f_L1.shp
COASTLINE_SHP_DEFAULT = r"F:/ZJU/11_Ice/dataset_create/data_apply/GSHHS_shp/GSHHS_shp/f/GSHHS_f_L1.shp"

# 可选：Antarctic 冰架层（把冰架排除出陆地，使其仍参与海冰分类）
# 为 None 时不做冰架排除
ANTARCTIC_ICE_SHELVES_SHP_DEFAULT = None

# 陆地掩膜膨胀像素数（缓冲潮汐 / 地理配准误差）。80m 分辨率下 1 像素约 80m。
LAND_BUFFER_PIXELS_DEFAULT = 1

# 断续重传
ENABLE_RESUME_STATE = True
RESUME_STATE_FILENAME = "pack_for_inference_resume_state.json"
MAX_RETRY_PER_SCENE = 2
RETRY_WAIT_SECONDS = 3
SKIP_EXISTING_NC = True

# SNAP 输出中可能出现的波段文件名候选（与 4_auto_make_labels.py 保持一致）
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

NODATA_VALUE = -9999.0


# ============================================================
# 基础工具
# ============================================================

def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"[计时] {func.__name__}: {end - start:.2f} s")
        return result
    return wrapper


def _now_iso():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def find_band_file(sar_data_dir, filename_candidates):
    for filename in filename_candidates:
        full_path = os.path.join(sar_data_dir, filename)
        if os.path.exists(full_path):
            return full_path.replace('\\', '/')
    return None


def find_glcm_files(sar_data_dir):
    pattern = os.path.join(sar_data_dir, "Sigma0_HH_*.img")
    glcm_paths = sorted(glob.glob(pattern))
    return [p.replace('\\', '/') for p in glcm_paths]


def build_sar_invalid_mask(hh_array, hv_array, inc_angle_array, hh_nodata, hv_nodata, inc_nodata):
    invalid_mask = (~np.isfinite(hh_array)) | (~np.isfinite(hv_array)) | (~np.isfinite(inc_angle_array))
    if hh_nodata is not None and np.isfinite(hh_nodata):
        invalid_mask |= (hh_array == hh_nodata)
    if hv_nodata is not None and np.isfinite(hv_nodata):
        invalid_mask |= (hv_array == hv_nodata)
    if inc_nodata is not None and np.isfinite(inc_nodata):
        invalid_mask |= (inc_angle_array == inc_nodata)
    invalid_mask |= (hh_array <= -9990.0) | (hv_array <= -9990.0) | (inc_angle_array <= -9990.0)
    return invalid_mask


# ============================================================
# 海岸线加载与栅格化
# ============================================================

def load_coastline_gdf(coastline_shp, antarctic_ice_shelves_shp=None):
    """
    加载海岸线（陆地）多边形；可选从陆地中扣除南极冰架（使冰架仍参与海冰分类）。
    返回 WGS84 (EPSG:4326) 下的 GeoDataFrame。
    """
    print(f"[海岸线] 加载 GSHHG: {coastline_shp}")
    if not os.path.exists(coastline_shp):
        raise FileNotFoundError(f"海岸线文件不存在: {coastline_shp}")

    land = gpd.read_file(coastline_shp)
    if land.crs is None:
        land = land.set_crs(epsg=4326)
    else:
        land = land.to_crs(epsg=4326)
    print(f"[海岸线] 陆地多边形数量: {len(land)}")

    if antarctic_ice_shelves_shp and os.path.exists(antarctic_ice_shelves_shp):
        print(f"[海岸线] 加载南极冰架层用于扣除: {antarctic_ice_shelves_shp}")
        shelves = gpd.read_file(antarctic_ice_shelves_shp).to_crs(epsg=4326)
        shelves_union = shelves.unary_union
        # 从陆地中减去冰架范围
        land['geometry'] = land.geometry.difference(shelves_union)
        land = land[~land.geometry.is_empty].reset_index(drop=True)
        print(f"[海岸线] 扣除冰架后陆地多边形数量: {len(land)}")

    return land


def rasterize_coastline_land_mask(land_gdf_wgs84, sar_shape, sar_transform, sar_crs, buffer_pixels=1):
    """
    在 SAR 的 CRS / 网格上生成陆地布尔掩膜。

    Parameters
    ----------
    land_gdf_wgs84 : GeoDataFrame  全球陆地多边形（EPSG:4326）
    sar_shape      : (H, W)
    sar_transform  : rasterio Affine
    sar_crs        : rasterio CRS
    buffer_pixels  : int  陆地向海侧膨胀的像素数（缓冲潮汐 / 配准误差）

    Returns
    -------
    land_mask : np.ndarray(bool), shape=sar_shape
    """
    h, w = sar_shape

    # 计算 SAR 场景在 WGS84 下的 BBox，先裁一次减少多边形数
    corners_px = [(0, 0), (w, 0), (0, h), (w, h)]
    xs, ys = [], []
    for cx, cy in corners_px:
        x, y = sar_transform * (cx, cy)
        xs.append(x); ys.append(y)
    # SAR CRS 下的 BBox（含一点余量）
    pad_x = 0.02 * (max(xs) - min(xs))
    pad_y = 0.02 * (max(ys) - min(ys))
    bbox_sar_crs = box(min(xs) - pad_x, min(ys) - pad_y,
                       max(xs) + pad_x, max(ys) + pad_y)

    # 把 BBox 投回 WGS84 以便裁全球陆地
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_sar_crs], crs=sar_crs).to_crs(epsg=4326)
    bbox_wgs84 = bbox_gdf.geometry.iloc[0]

    # 用空间索引裁 (cx 用 bbox 快速过滤)
    minx, miny, maxx, maxy = bbox_wgs84.bounds
    candidate = land_gdf_wgs84.cx[minx:maxx, miny:maxy]
    if len(candidate) == 0:
        print("[海岸线] 场景 BBox 内无陆地多边形，返回全零掩膜")
        return np.zeros(sar_shape, dtype=bool)
    print(f"[海岸线] BBox 内陆地多边形数量: {len(candidate)}")

    # 重投影到 SAR CRS 后栅格化
    candidate_sar = candidate.to_crs(sar_crs)
    shapes_iter = ((geom, 1) for geom in candidate_sar.geometry if geom is not None and not geom.is_empty)
    land_mask = features.rasterize(
        shapes=shapes_iter,
        out_shape=sar_shape,
        transform=sar_transform,
        fill=0,
        dtype='uint8',
    ).astype(bool)

    # 膨胀缓冲
    if buffer_pixels and buffer_pixels > 0:
        try:
            from scipy.ndimage import binary_dilation
            land_mask = binary_dilation(land_mask, iterations=int(buffer_pixels))
        except ImportError:
            print("[海岸线] 未安装 scipy，跳过膨胀缓冲")

    land_ratio = float(land_mask.sum()) / float(land_mask.size)
    print(f"[海岸线] 陆地像素占比: {land_ratio:.2%}")
    return land_mask


# ============================================================
# NC 打包
# ============================================================

def package_inference_to_netcdf(
    hh_array, hv_array, inc_angle_array,
    glcm_data_dict, coastline_land_mask,
    sar_transform, sar_crs,
    output_nc_path,
):
    print(f"[打包] 构建 xarray.Dataset -> {output_nc_path}")
    dims = ["y", "x"]
    data_vars = {
        "nersc_sar_primary":   (dims, hh_array.astype(np.float32),        {"description": "Sigma0 in dB", "polarisation": "HH"}),
        "nersc_sar_secondary": (dims, hv_array.astype(np.float32),        {"description": "Sigma0 in dB", "polarisation": "HV"}),
        "sar_incidenceangle":  (dims, inc_angle_array.astype(np.float32), {"description": "Incidence angle", "units": "degrees"}),
        "coastline_land_mask": (dims, coastline_land_mask.astype(np.uint8),
                                {"description": "Land mask from coastline vector (e.g. GSHHG)",
                                 "flag_values": np.array([0, 1], dtype=np.uint8),
                                 "flag_meanings": "sea land"}),
    }
    for var_name, var_array in glcm_data_dict.items():
        data_vars[var_name] = (
            dims,
            var_array.astype(np.float32),
            {"description": "GLCM texture feature from Sigma0_HH_corrected"},
        )

    # 将 affine 写成 6 浮点（GDAL 风格：a, b, c, d, e, f）
    geo_transform = [float(v) for v in (
        sar_transform.a, sar_transform.b, sar_transform.c,
        sar_transform.d, sar_transform.e, sar_transform.f,
    )]
    crs_wkt = sar_crs.to_wkt() if sar_crs is not None else ""

    attrs = {
        "description": "Inference-ready SAR package (no icechart labels)",
        "source": "4b_pack_for_inference.py",
        "created_at": _now_iso(),
        "crs_wkt": crs_wkt,
        "geo_transform": geo_transform,
        "land_mask_source": "coastline vector",
        "invalid_fill_value": float(NODATA_VALUE),
    }

    ds = xr.Dataset(data_vars=data_vars, attrs=attrs)

    chunk_size = (512, 512)
    encoding_dict = {}
    for var in ds.data_vars:
        enc = {"zlib": True, "complevel": 4, "shuffle": True, "chunksizes": chunk_size}
        encoding_dict[var] = enc

    os.makedirs(os.path.dirname(output_nc_path), exist_ok=True)
    ds.to_netcdf(output_nc_path, mode='w', format='NETCDF4', engine='h5netcdf', encoding=encoding_dict)
    print(f"[打包] 完成: {output_nc_path}")


# ============================================================
# 单场景处理
# ============================================================

def process_single_scene_for_inference(sar_dir, land_gdf_wgs84, output_nc_path, buffer_pixels):
    hh_img_path = find_band_file(sar_dir, HH_FILENAME_CANDIDATES)
    hv_img_path = find_band_file(sar_dir, HV_FILENAME_CANDIDATES)
    inc_angle_path = find_band_file(sar_dir, INC_FILENAME_CANDIDATES)

    missing = []
    if hh_img_path is None:    missing.append("HH")
    if hv_img_path is None:    missing.append("HV")
    if inc_angle_path is None: missing.append("incidence angle")
    if missing:
        raise FileNotFoundError(f"场景缺少必要波段文件: {', '.join(missing)}")

    # 1) 读取 SAR 空间参考
    print(f"[1/5] 读取 SAR 空间信息: {hh_img_path}")
    with rasterio.open(hh_img_path) as src:
        sar_shape = (src.height, src.width)
        sar_transform = src.transform
        sar_crs = src.crs
        hh_array = src.read(1)
        hh_nodata = src.nodata
    if sar_crs is None:
        raise RuntimeError("SAR 文件缺少 CRS，无法对齐海岸线。请检查 SNAP Terrain Correction 是否成功。")

    # 2) 读取 HV / 角度 / GLCM
    print(f"[2/5] 读取 SAR 特征波段")
    with rasterio.open(hv_img_path) as src:
        hv_array = src.read(1)
        hv_nodata = src.nodata
    with rasterio.open(inc_angle_path) as src:
        inc_angle_array = src.read(1)
        inc_nodata = src.nodata

    glcm_data_dict = {}
    glcm_paths = find_glcm_files(sar_dir)
    print(f"[2/5] 检测到 GLCM 波段: {len(glcm_paths)}")
    for glcm_path in glcm_paths:
        glcm_name = os.path.splitext(os.path.basename(glcm_path))[0]
        var_name = f"glcm_{glcm_name.replace('Sigma0_HH_corrected_', '').lower()}"
        with rasterio.open(glcm_path) as src:
            glcm_data_dict[var_name] = src.read(1)
        if glcm_data_dict[var_name].shape != sar_shape:
            raise ValueError(f"GLCM 波段尺寸与 HH 不一致: {var_name}, "
                             f"{glcm_data_dict[var_name].shape} vs {sar_shape}")

    # 3) 构建雷达无效掩膜
    print(f"[3/5] 构建雷达无效掩膜")
    invalid_mask = build_sar_invalid_mask(
        hh_array, hv_array, inc_angle_array,
        hh_nodata, hv_nodata, inc_nodata,
    )

    # 4) 栅格化海岸线 → 陆地掩膜
    print(f"[4/5] 栅格化海岸线")
    coastline_land_mask = rasterize_coastline_land_mask(
        land_gdf_wgs84, sar_shape, sar_transform, sar_crs,
        buffer_pixels=buffer_pixels,
    )

    # 5) 合并最终掩膜 → 置 -9999 → 打包
    final_mask = invalid_mask | coastline_land_mask
    total = int(final_mask.size)
    print(f"[掩膜检查] 雷达无效 {invalid_mask.sum()}/{total} ({invalid_mask.sum()/total:.2%}) | "
          f"陆地 {coastline_land_mask.sum()}/{total} ({coastline_land_mask.sum()/total:.2%}) | "
          f"合并 {final_mask.sum()}/{total} ({final_mask.sum()/total:.2%})")

    hh_array[final_mask] = NODATA_VALUE
    hv_array[final_mask] = NODATA_VALUE
    inc_angle_array[final_mask] = NODATA_VALUE
    for var_name, arr in glcm_data_dict.items():
        arr[final_mask] = NODATA_VALUE

    print(f"[5/5] 打包 NC")
    package_inference_to_netcdf(
        hh_array, hv_array, inc_angle_array,
        glcm_data_dict, coastline_land_mask,
        sar_transform, sar_crs,
        output_nc_path,
    )


# ============================================================
# 断续重传
# ============================================================

def get_resume_state_path(output_nc_dir):
    return os.path.join(output_nc_dir, RESUME_STATE_FILENAME).replace('\\', '/')


def load_resume_state(state_path):
    default_state = {
        "meta": {"created_at": _now_iso(), "updated_at": _now_iso(), "state_version": 1},
        "scenes": {},
    }
    if not ENABLE_RESUME_STATE or not os.path.exists(state_path):
        return default_state
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        if not isinstance(state, dict):          return default_state
        if "meta" not in state:                  state["meta"] = default_state["meta"]
        if "scenes" not in state:                state["scenes"] = {}
        return state
    except Exception as e:
        print(f"[断续重传] 状态文件读取失败: {e}")
        return default_state


def save_resume_state(state_path, state):
    if not ENABLE_RESUME_STATE:
        return
    state["meta"]["updated_at"] = _now_iso()
    tmp = f"{state_path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, state_path)


def update_scene_state(state, scene_name, **kwargs):
    info = state["scenes"].get(scene_name, {})
    info.update(kwargs)
    info["updated_at"] = _now_iso()
    state["scenes"][scene_name] = info


def get_scene_attempts(state, scene_name):
    info = state["scenes"].get(scene_name, {})
    attempts = info.get("attempts", 0)
    return attempts if isinstance(attempts, int) and attempts >= 0 else 0


# ============================================================
# 主流程
# ============================================================

@timer_decorator
def run(sar_root, output_nc_dir, coastline_shp, antarctic_shelves_shp, buffer_pixels):
    print("====== 推理包装流水线启动（无冰图）======")
    os.makedirs(output_nc_dir, exist_ok=True)

    sar_dirs = sorted(glob.glob(os.path.join(sar_root, "*_glcm.data")))
    if len(sar_dirs) == 0:
        print(f"未找到 SAR 场景（期望形如 *_glcm.data）: {sar_root}")
        return
    print(f"检测到 SAR 场景数: {len(sar_dirs)}")

    # 一次性加载全球陆地多边形，供所有场景复用
    land_gdf = load_coastline_gdf(coastline_shp, antarctic_shelves_shp)

    state_path = get_resume_state_path(output_nc_dir)
    resume_state = load_resume_state(state_path)
    if ENABLE_RESUME_STATE:
        print(f"[断续重传] 状态文件: {state_path} | 已记录场景数: {len(resume_state['scenes'])}")

    success = 0
    skipped_existing = 0
    resumed_skip = 0
    exhausted = 0
    failed = 0

    for idx, sar_dir in enumerate(sar_dirs, start=1):
        sar_name = os.path.basename(sar_dir)
        print("\n" + "=" * 88)
        print(f"[{idx}/{len(sar_dirs)}] 处理: {sar_name}")

        scene_prefix = sar_name.replace("_glcm.data", "")
        output_nc_path = os.path.join(output_nc_dir, f"{scene_prefix}.nc").replace('\\', '/')

        scene_state = resume_state["scenes"].get(sar_name, {})
        if ENABLE_RESUME_STATE and scene_state.get("status") == "success" and os.path.exists(output_nc_path):
            resumed_skip += 1
            print(f"  -> 断续重传跳过: {output_nc_path}")
            continue

        if SKIP_EXISTING_NC and os.path.exists(output_nc_path):
            skipped_existing += 1
            print(f"  -> 跳过：目标 NC 已存在: {output_nc_path}")
            update_scene_state(resume_state, sar_name,
                               status="skipped_existing",
                               attempts=get_scene_attempts(resume_state, sar_name),
                               output_nc_path=output_nc_path, last_error="")
            save_resume_state(state_path, resume_state)
            continue

        max_total = 1 + MAX_RETRY_PER_SCENE
        prev = get_scene_attempts(resume_state, sar_name)
        if prev >= max_total:
            exhausted += 1
            print(f"  -> 跳过：历史失败次数已达上限 {prev}/{max_total}")
            continue

        scene_success = False
        for attempt in range(prev + 1, max_total + 1):
            print(f"  -> 尝试 {attempt}/{max_total}")
            update_scene_state(resume_state, sar_name,
                               status="running", attempts=attempt,
                               output_nc_path=output_nc_path, last_error="")
            save_resume_state(state_path, resume_state)
            try:
                process_single_scene_for_inference(sar_dir, land_gdf, output_nc_path, buffer_pixels)
                success += 1
                scene_success = True
                update_scene_state(resume_state, sar_name,
                                   status="success", attempts=attempt,
                                   output_nc_path=output_nc_path, last_error="")
                save_resume_state(state_path, resume_state)
                print(f"  -> 成功: {output_nc_path}")
                break
            except Exception as e:
                err = str(e)
                print(f"  -> 失败: {err}")
                update_scene_state(resume_state, sar_name,
                                   status="failed", attempts=attempt,
                                   output_nc_path=output_nc_path, last_error=err)
                save_resume_state(state_path, resume_state)
                if attempt < max_total:
                    print(f"  -> {RETRY_WAIT_SECONDS} 秒后重试...")
                    time.sleep(RETRY_WAIT_SECONDS)

        if not scene_success:
            failed += 1

    print("\n" + "=" * 88)
    print(f"完成：成功 {success} | 已存在跳过 {skipped_existing} | 续跑跳过 {resumed_skip} | "
          f"达上限 {exhausted} | 失败 {failed} | 总 {len(sar_dirs)}")


def parse_args():
    p = argparse.ArgumentParser(description="将 SNAP 处理后的 SAR 打包为推理用 NC（无冰图，含海岸线陆地掩膜）")
    p.add_argument("--sar_root", default=SAR_ROOT_DIR_DEFAULT,
                   help="包含 *_glcm.data/ 子目录的 SNAP 输出根目录")
    p.add_argument("--out", default=OUTPUT_NC_DIR_DEFAULT,
                   help="输出 NC 目录")
    p.add_argument("--coastline", default=COASTLINE_SHP_DEFAULT,
                   help="陆地矢量 shp 路径（推荐 GSHHG full: GSHHS_f_L1.shp）")
    p.add_argument("--antarctic_shelves", default=ANTARCTIC_ICE_SHELVES_SHP_DEFAULT,
                   help="可选：从陆地中扣除的南极冰架 shp 路径")
    p.add_argument("--buffer_pixels", type=int, default=LAND_BUFFER_PIXELS_DEFAULT,
                   help="陆地掩膜膨胀像素数（默认 1，80m 分辨率约 80m 缓冲）")
    return p.parse_args()


def main():
    args = parse_args()
    run(
        sar_root=args.sar_root,
        output_nc_dir=args.out,
        coastline_shp=args.coastline,
        antarctic_shelves_shp=args.antarctic_shelves,
        buffer_pixels=args.buffer_pixels,
    )


if __name__ == "__main__":
    main()
