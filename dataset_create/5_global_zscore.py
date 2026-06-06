import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr



RUN_MODE = "fit_apply"  # 可选: fit, apply, fit_apply
NC_ROOT = r"F:/ZJU/11_Ice/dataset_create/dataset_nc_new"
OUTPUT_ROOT = r"F:/ZJU/11_Ice/dataset_create/dataset_nc_zscore_new"

STATS_JSON = None  # None 时默认使用 NC_ROOT/zscore_stats.json
VARIABLES = None  # None 时自动选择 SAR + GLCM 特征
NODATA_VALUE = -9999.0
EPS_VALUE = 1e-6
TRAIN_SAFE_OUTPUT = True  # True: 额外输出 valid mask，并将无效值映射为 0
SKIP_BROKEN_NC = True  # True: 遇到损坏/不可读 nc 时跳过并继续
BROKEN_FILES_LOG = None  # None 时默认写入 NC_ROOT/zscore_broken_files.json
BROKEN_FILENAMES_DOC = None  # None 时默认写入 NC_ROOT/zscore_broken_filenames.txt

# 仅对 SAR dB 通道做固定截断，其他变量不截断。
CLIP_BOUNDS = {
    "nersc_sar_primary": (-35.0, 5.0),
    "nersc_sar_secondary": (-35.0, 5.0),
}



def list_dataset_files(root: str) -> List[str]:
    files = []
    for ext in ("*.nc",):
        files.extend(glob.glob(os.path.join(root, ext)))
    return sorted([p.replace("\\", "/") for p in files])


def auto_pick_feature_variables(ds: xr.Dataset) -> List[str]:
    picked = []
    for var in ds.data_vars:
        if var in ["nersc_sar_primary", "nersc_sar_secondary", "sar_incidenceangle"]:
            picked.append(var)
        elif var.startswith("glcm_"):
            picked.append(var)
    return picked


def valid_mask_for_array(values: np.ndarray, nodata: float, fill_value: Optional[float]) -> np.ndarray:
    mask = np.isfinite(values)
    mask &= (values != nodata)
    if fill_value is not None and np.isfinite(fill_value):
        mask &= (values != float(fill_value))
    return mask


def sanitize_attrs_for_netcdf(attrs: Dict) -> Dict:
    sanitized = {}
    for key, value in attrs.items():
        if isinstance(value, (bool, np.bool_)):
            sanitized[key] = np.int8(1 if value else 0)
        elif isinstance(value, np.ndarray) and value.dtype == np.bool_:
            sanitized[key] = value.astype(np.int8)
        else:
            sanitized[key] = value
    return sanitized


def clip_if_configured(values: np.ndarray, var_name: str) -> np.ndarray:
    bounds = CLIP_BOUNDS.get(var_name)
    if bounds is None:
        return values
    low, high = bounds
    return np.clip(values, a_min=low, a_max=high)


def build_global_valid_mask(ds: xr.Dataset, variables: List[str], nodata: float) -> Optional[np.ndarray]:
    base_shape = None
    global_valid_mask = None

    for var in variables:
        if var not in ds.data_vars:
            continue

        arr = np.asarray(ds[var].values)
        if base_shape is None:
            base_shape = arr.shape
            global_valid_mask = np.ones(base_shape, dtype=bool)
        elif arr.shape != base_shape:
            raise ValueError(f"变量尺寸不一致，无法构建全局掩膜: {var}, shape={arr.shape}, base={base_shape}")

        fill_value = ds[var].attrs.get("_FillValue", None)
        var_valid = valid_mask_for_array(arr, nodata, fill_value)
        global_valid_mask &= var_valid

    return global_valid_mask


def fit_global_stats(nc_files: List[str], variables: List[str], nodata: float) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, str]]]:
    stats = {
        var: {
            "count": 0,
            "sum": 0.0,
            "sumsq": 0.0,
        }
        for var in variables
    }
    broken_files: List[Dict[str, str]] = []

    for idx, path in enumerate(nc_files, start=1):
        print(f"[fit] {idx}/{len(nc_files)}: {os.path.basename(path)}")
        try:
            with xr.open_dataset(path, engine="h5netcdf") as ds:
                global_valid_mask = build_global_valid_mask(ds, variables, nodata)
                if global_valid_mask is None:
                    continue

                for var in variables:
                    if var not in ds:
                        continue
                    arr = np.asarray(ds[var].values, dtype=np.float64)
                    vals = arr[global_valid_mask]
                    if vals.size == 0:
                        continue

                    vals = clip_if_configured(vals, var)

                    stats[var]["count"] += int(vals.size)
                    stats[var]["sum"] += float(vals.sum(dtype=np.float64))
                    stats[var]["sumsq"] += float(np.square(vals, dtype=np.float64).sum(dtype=np.float64))
        except Exception as e:
            if SKIP_BROKEN_NC:
                print(f"[fit][skip-broken] {os.path.basename(path)}: {e}")
                broken_files.append({"path": path, "stage": "fit", "error": str(e)})
                continue
            raise

    result: Dict[str, Dict[str, float]] = {}
    for var in variables:
        count = stats[var]["count"]
        if count <= 0:
            result[var] = {"count": 0, "mean": float("nan"), "std": float("nan")}
            continue

        mean = stats[var]["sum"] / count
        var_pop = max(stats[var]["sumsq"] / count - mean * mean, 0.0)
        std = float(np.sqrt(var_pop))
        result[var] = {
            "count": int(count),
            "mean": float(mean),
            "std": std,
        }
    return result, broken_files


def save_broken_files_log(broken_files: List[Dict[str, str]], output_json: str) -> None:
    if len(broken_files) == 0:
        return

    out_dir = os.path.dirname(output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    payload = {
        "count": len(broken_files),
        "broken_files": broken_files,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"坏文件日志已保存: {output_json}")


def save_broken_filenames_doc(broken_files: List[Dict[str, str]], output_doc: str) -> None:
    """将坏文件名写入文档（每行一个文件名，去重并排序）。"""
    if len(broken_files) == 0:
        return

    out_dir = os.path.dirname(output_doc)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    filename_set = set()
    for item in broken_files:
        path = item.get("path", "")
        if isinstance(path, str) and len(path) > 0:
            filename_set.add(os.path.basename(path))

    with open(output_doc, "w", encoding="utf-8") as f:
        f.write("# 坏文件名列表\n")
        f.write(f"总数(去重后): {len(filename_set)}\n\n")
        for name in sorted(filename_set):
            f.write(f"{name}\n")

    print(f"坏文件名文档已保存: {output_doc}")


def save_stats(stats: Dict[str, Dict[str, float]], output_json: str, nc_root: str, variables: List[str], nodata: float) -> None:
    payload = {
        "nc_root": nc_root,
        "nodata": nodata,
        "variables": variables,
        "stats": stats,
    }
    out_dir = os.path.dirname(output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"统计量已保存: {output_json}")


def load_stats(stats_json: str) -> Dict[str, Dict[str, float]]:
    with open(stats_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "stats" not in payload:
        raise ValueError("统计文件格式错误：缺少 stats 字段")
    return payload["stats"]


def normalize_one_dataset(
    ds: xr.Dataset,
    stats: Dict[str, Dict[str, float]],
    variables: List[str],
    nodata: float,
    eps: float,
    train_safe_output: bool,
) -> xr.Dataset:
    new_data_vars = {}

    base_dims = None
    for var in variables:
        if var in ds.data_vars:
            base_dims = ds[var].dims
            break

    if base_dims is None:
        return ds  # 没有需要处理的变量，直接返回

    global_valid_mask = build_global_valid_mask(ds, variables, nodata)
    if global_valid_mask is None:
        return ds

    for var in ds.data_vars:
        da = ds[var]
        attrs = sanitize_attrs_for_netcdf(dict(da.attrs))
        arr = np.asarray(da.values)

        if var in variables and var in stats and int(stats[var].get("count", 0)) > 0:
            mean = float(stats[var]["mean"])
            std = float(stats[var]["std"])
            denom = std if std > eps else eps

            arr_float = arr.astype(np.float32, copy=False)
            out = arr_float.copy()
            
            arr_float[global_valid_mask] = clip_if_configured(arr_float[global_valid_mask], var)

            # 直接使用刚刚算好的 global_valid_mask
            out[global_valid_mask] = (arr_float[global_valid_mask] - mean) / denom
            out[~global_valid_mask] = 0.0 if train_safe_output else nodata

            attrs.pop("_FillValue", None)
            attrs["zscore_mean"] = mean
            attrs["zscore_std"] = std
            attrs["zscore_applied"] = np.int8(1)
            if train_safe_output:
                attrs["invalid_value_mapped_to"] = 0.0
                attrs["training_safe_output"] = np.int8(1)
                
            new_data_vars[var] = (da.dims, out.astype(np.float32), attrs)
        else:
            # 标签等不需要归一化的变量，直接原样保留
            new_data_vars[var] = (da.dims, arr, attrs)

    # ================= 新增：统一写入一个全局掩膜图层 =================
    if train_safe_output:
        mask_attrs = {
            "long_name": "Global valid mask for all input features",
            "flag_values": np.array([0, 1], dtype=np.uint8),
            "flag_meanings": "invalid valid",
        }
        new_data_vars["global_valid_mask"] = (base_dims, global_valid_mask.astype(np.uint8), mask_attrs)

    # 更新全局元数据
    new_attrs = sanitize_attrs_for_netcdf(dict(ds.attrs))
    new_attrs["normalization"] = "global_zscore"
    new_attrs["normalization_nodata"] = nodata
    new_attrs["training_safe_output"] = np.int8(1 if train_safe_output else 0)
    if train_safe_output:
        new_attrs["training_safe_mask_layer"] = "global_valid_mask" # 元数据说明掩膜的名字
        new_attrs["training_safe_invalid_mapped_to"] = 0.0

    return xr.Dataset(data_vars=new_data_vars, coords=ds.coords, attrs=new_attrs)


def build_encoding(ds: xr.Dataset, variables: List[str], nodata: float, train_safe_output: bool) -> Dict[str, Dict]:
    chunk_size = (512, 512)
    encoding = {}
    for var in ds.data_vars:
        base = {"zlib": True, "complevel": 4, "shuffle": True, "chunksizes": chunk_size}
        
        # 针对全局唯一掩膜的压缩编码设置
        if train_safe_output and var == "global_valid_mask":
            base["dtype"] = "uint8"
            base["_FillValue"] = np.uint8(0)
            
        elif var in variables:
            base["dtype"] = "float32"
            base["_FillValue"] = np.float32(nodata)
            
        encoding[var] = base
    return encoding

def apply_global_zscore(
    nc_files: List[str],
    output_root: str,
    stats: Dict[str, Dict[str, float]],
    variables: List[str],
    nodata: float,
    eps: float,
    train_safe_output: bool,
) -> List[Dict[str, str]]:
    os.makedirs(output_root, exist_ok=True)
    broken_files: List[Dict[str, str]] = []

    for idx, path in enumerate(nc_files, start=1):
        out_path = os.path.join(output_root, os.path.basename(path)).replace("\\", "/")
        print(f"[apply] {idx}/{len(nc_files)}: {os.path.basename(path)}")

        try:
            with xr.open_dataset(path, engine="h5netcdf") as ds:
                out_ds = normalize_one_dataset(ds, stats, variables, nodata, eps, train_safe_output)
                encoding = build_encoding(out_ds, variables, nodata, train_safe_output)
                out_ds.to_netcdf(
                    out_path,
                    mode="w",
                    format="NETCDF4",
                    engine="h5netcdf",
                    encoding=encoding,
                )
        except Exception as e:
            if SKIP_BROKEN_NC:
                print(f"[apply][skip-broken] {os.path.basename(path)}: {e}")
                broken_files.append({"path": path, "stage": "apply", "error": str(e)})
                continue
            raise

    print(f"归一化完成，输出目录: {output_root}")
    return broken_files


def main() -> None:
    mode = RUN_MODE
    if mode not in ["fit", "apply", "fit_apply"]:
        raise ValueError(f"RUN_MODE 非法: {mode}，应为 fit/apply/fit_apply")

    nc_root = NC_ROOT.replace("\\", "/")
    output_root = OUTPUT_ROOT.replace("\\", "/")

    nc_files = list_dataset_files(nc_root)
    if len(nc_files) == 0:
        raise FileNotFoundError(f"目录下未找到 .nc 文件: {nc_root}")

    default_stats_json = os.path.join(nc_root, "zscore_stats.json").replace("\\", "/")
    stats_json = STATS_JSON.replace("\\", "/") if STATS_JSON else default_stats_json
    default_broken_log = os.path.join(nc_root, "zscore_broken_files.json").replace("\\", "/")
    broken_log_json = BROKEN_FILES_LOG.replace("\\", "/") if BROKEN_FILES_LOG else default_broken_log
    default_broken_doc = os.path.join(nc_root, "zscore_broken_filenames.txt").replace("\\", "/")
    broken_filenames_doc = BROKEN_FILENAMES_DOC.replace("\\", "/") if BROKEN_FILENAMES_DOC else default_broken_doc

    with xr.open_dataset(nc_files[0], engine="h5netcdf") as sample_ds:
        variables = VARIABLES if VARIABLES else auto_pick_feature_variables(sample_ds)

    if len(variables) == 0:
        raise ValueError("未找到可归一化变量，请通过 --variables 指定")

    print(f"文件数: {len(nc_files)}")
    print(f"归一化变量: {variables}")
    print(f"mode: {mode}")
    print(f"training_safe_output: {TRAIN_SAFE_OUTPUT}")

    stats: Optional[Dict[str, Dict[str, float]]] = None
    all_broken_files: List[Dict[str, str]] = []

    if mode in ["fit", "fit_apply"]:
        stats, broken_fit = fit_global_stats(nc_files, variables, NODATA_VALUE)
        all_broken_files.extend(broken_fit)
        save_stats(stats, stats_json, nc_root, variables, NODATA_VALUE)

    if mode in ["apply", "fit_apply"]:
        if stats is None:
            if not os.path.exists(stats_json):
                raise FileNotFoundError(f"找不到统计文件: {stats_json}")
            stats = load_stats(stats_json)
        broken_apply = apply_global_zscore(
            nc_files,
            output_root,
            stats,
            variables,
            NODATA_VALUE,
            EPS_VALUE,
            TRAIN_SAFE_OUTPUT,
        )
        all_broken_files.extend(broken_apply)

    if SKIP_BROKEN_NC and len(all_broken_files) > 0:
        # 同一路径在 fit/apply 都失败时会出现两条记录，保留 stage 信息便于排查。
        save_broken_files_log(all_broken_files, broken_log_json)
        save_broken_filenames_doc(all_broken_files, broken_filenames_doc)
        print(f"坏文件数量: {len(all_broken_files)}（详情见日志）")


if __name__ == "__main__":
    main()