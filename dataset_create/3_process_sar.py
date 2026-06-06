import os
import sys
import time
import json
from functools import wraps
import subprocess
import shutil

proxy = 'http://127.0.0.1:7897' 
os.environ['http_proxy'] = proxy
os.environ['https_proxy'] = proxy

# SNAP gpt 命令行工具的路径
GPT_PATH = r"F:/Software/esa-snap/bin/gpt.exe"

# 原始下载的 S1 压缩包目录
INPUT_DIR = r"F:\ZJU\11_Ice\dataset_create\data_apply\S1_raw"

# 输出工作目录
OUTPUT_DIR = r"F:\ZJU\11_Ice\dataset_create\data_apply\S1_process"

# 第4步 NC 文件可能存在的多个输出目录（任意一个中找到 {base_name}.nc 即跳过该场景）
OUTPUT_NC_DIRS = [
    r"F:/ZJU/11_Ice/dataset_create/data_nc_CAA",    # 目录1
    r"F:/ZJU/11_Ice/dataset_create/data_nc_24_2-4", # 目录2
    r"F:/ZJU/11_Ice/dataset_create/data_nc_cls34",  # 目录3
    r"F:/ZJU/11_Ice/dataset_create/data_nc_hb",     # 目录4
    r"F:/ZJU/11_Ice/dataset_create/data_nc_wa",     # 目录5
    r"F:/ZJU/11_Ice/dataset_create/dataset_nc_new", # 目录6
]

# 断续重传状态文件（用于中断后继续）
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "process_progress.json").replace('\\', '/')

# 是否保留中间文件（True: 保留；False: 在 GLCM 成功后自动删除中间文件）
KEEP_INTERMEDIATE = False

# 如果输出目录不存在，则创建
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def build_paths(input_zip):
    base_name = os.path.splitext(os.path.basename(input_zip))[0]
    final_dim = os.path.join(OUTPUT_DIR, f"{base_name}_final.dim").replace('\\', '/')
    prep_xml = os.path.join(OUTPUT_DIR, f"{base_name}_prep_graph.xml").replace('\\', '/')
    glcm_dim = os.path.join(OUTPUT_DIR, f"{base_name}_glcm.dim").replace('\\', '/')
    glcm_xml = os.path.join(OUTPUT_DIR, f"{base_name}_glcm_graph.xml").replace('\\', '/')
    return {
        "base_name": base_name,
        "final_dim": final_dim,
        "prep_xml": prep_xml,
        "glcm_dim": glcm_dim,
        "glcm_xml": glcm_xml,
    }

def list_input_zips(input_dir):
    if not os.path.isdir(input_dir):
        print(f"输入目录不存在: {input_dir}")
        sys.exit(1)
    zip_files = [
        os.path.join(input_dir, name).replace('\\', '/')
        for name in os.listdir(input_dir)
        if name.lower().endswith('.zip')
    ]
    zip_files.sort()
    if not zip_files:
        print(f"在输入目录中未找到 .zip 文件: {input_dir}")
        sys.exit(1)
    return zip_files


def remove_path(path):
    if os.path.isdir(path):
      shutil.rmtree(path)
      print(f"已删除目录: {path}")
    elif os.path.isfile(path):
      os.remove(path)
      print(f"已删除文件: {path}")


def cleanup_after_glcm(paths):
    final_data_dir = paths['final_dim'].replace('.dim', '.data')
    final_hdr = paths['final_dim'].replace('.dim', '.hdr')

    remove_path(paths['final_dim'])
    remove_path(final_data_dir)
    remove_path(final_hdr)

    remove_path(paths['prep_xml'])
    remove_path(paths['glcm_xml'])


def is_unreadable_s1_zip_error(error_text):
    """识别 SNAP 无法读取 Sentinel-1 压缩包的典型报错。"""
    if not error_text:
      return False
  # 不同 SNAP 版本输出会有差异，尽量宽松匹配这一类“ZIP 结构/内容不可读”错误。
    signatures = [
      "Sentinel1ProductReader",
      "VirtualDir.getInputStream",
      "AbstractProductDirectory.getProductDir() is null",
      "NodeId: Read",
      "Cannot invoke",
    ] 
    hit_count = sum(sig in error_text for sig in signatures)
    return hit_count >= 2


def beam_dim_ready(dim_path):
    """检查 BEAM-DIMAP 产物是否完整（.dim + .data）。"""
    data_dir = dim_path.replace('.dim', '.data')
    return os.path.isfile(dim_path) and os.path.isdir(data_dir)


def load_progress():
    if not os.path.isfile(PROGRESS_FILE):
        return {}
    try:
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as e:
        print(f"警告: 读取进度文件失败，将重建进度文件。原因: {e}")
    return {}


def save_progress(progress):
    tmp_file = f"{PROGRESS_FILE}.tmp"
    with open(tmp_file, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)
    os.replace(tmp_file, PROGRESS_FILE)


def update_progress(progress, base_name, status, message=''):
    progress[base_name] = {
        'status': status,
        'updated_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        'message': message,
    }
    save_progress(progress)

# =========================================================================
# Graph 1: 预处理图 (步骤 1 到 8)
# 顺序: Orbit -> Thermal Noise -> Border Noise -> Calib -> Speckle -> Terrain -> dB -> Incidence Angle
# =========================================================================
PREP_GRAPH_TEMPLATE = """<graph id="PrepGraph">
  <version>1.0</version>
  <node id="Read">
    <operator>Read</operator>
    <parameters>
        <file>{input_zip}</file>
        <formatName>SENTINEL-1</formatName> 
    </parameters>
  </node>
  
  <node id="Apply-Orbit-File">
    <operator>Apply-Orbit-File</operator>
    <sources><sourceProduct refid="Read"/></sources>
    <parameters>
      <orbitType>Sentinel Precise (Auto Download)</orbitType>
      <polyDegree>3</polyDegree>
      <continueOnFail>false</continueOnFail>
    </parameters>
  </node>

  <node id="ThermalNoiseRemoval">
    <operator>ThermalNoiseRemoval</operator>
    <sources><sourceProduct refid="Apply-Orbit-File"/></sources>
    <parameters>
      <selectedPolarisations>HH,HV</selectedPolarisations>
      <removeThermalNoise>true</removeThermalNoise>
    </parameters>
  </node>

  <node id="Remove-GRD-Border-Noise">
    <operator>Remove-GRD-Border-Noise</operator>
    <sources><sourceProduct refid="ThermalNoiseRemoval"/></sources>
    <parameters>
      <selectedPolarisations>HH,HV</selectedPolarisations>
      <borderLimit>500</borderLimit>
      <trimThreshold>0.5</trimThreshold>
    </parameters>
  </node>
  
  <node id="Calibration">
    <operator>Calibration</operator>
    <sources><sourceProduct refid="Remove-GRD-Border-Noise"/></sources>
    <parameters>      
      <outputImageInComplex>false</outputImageInComplex>
      <outputImageScaleInDb>false</outputImageScaleInDb>
      <selectedPolarisations>HH,HV</selectedPolarisations>
      <outputSigmaBand>true</outputSigmaBand>
    </parameters>
  </node>

  <node id="Speckle-Filter">
    <operator>Speckle-Filter</operator>
    <sources><sourceProduct refid="Calibration"/></sources>
    <parameters>
      <sourceBands>Sigma0_HH,Sigma0_HV</sourceBands>
      <filter>Refined Lee</filter>
      <filterSizeX>3</filterSizeX>
      <filterSizeY>3</filterSizeY>
      <estimateENL>true</estimateENL>
      <windowSize>7x7</windowSize>
      <targetWindowSizeStr>3x3</targetWindowSizeStr>
    </parameters>
  </node>
  
  <node id="Terrain-Correction">
    <operator>Terrain-Correction</operator>
    <sources><sourceProduct refid="Speckle-Filter"/></sources>
    <parameters>
      <sourceBands>Sigma0_HH,Sigma0_HV</sourceBands>
      <demName>Copernicus 90m Global DEM</demName>
      <demResamplingMethod>BILINEAR_INTERPOLATION</demResamplingMethod>
      <imgResamplingMethod>BILINEAR_INTERPOLATION</imgResamplingMethod>
      <pixelSpacingInMeter>80.0</pixelSpacingInMeter>
      <nodataValueAtSea>false</nodataValueAtSea>
      
      <saveIncidenceAngleFromEllipsoid>true</saveIncidenceAngleFromEllipsoid>
      <saveLocalIncidenceAngle>true</saveLocalIncidenceAngle>
    </parameters>
  </node>
  
  <node id="LinearToFromdB">
    <operator>LinearToFromdB</operator>
    <sources><sourceProduct refid="Terrain-Correction"/></sources>
    <parameters>
      <sourceBands>Sigma0_HH,Sigma0_HV</sourceBands>
    </parameters>
  </node>

  <node id="MergeAnglesAndDB">
    <operator>BandMerge</operator>
    <sources>
      <sourceProduct refid="LinearToFromdB"/>
      <sourceProduct.1 refid="Terrain-Correction"/>
    </sources>
  </node>

  <node id="BandMaths">
    <operator>BandMaths</operator>
    <sources>
      <sourceProduct refid="MergeAnglesAndDB"/>
    </sources>
    <parameters>
      <targetBands>
        <targetBand>
          <name>Sigma0_HH</name>
          <type>float32</type>
          <expression>(Sigma0_HH == 0.0) ? -9999.0 : Sigma0_HH_db</expression>
          <noDataValue>-9999.0</noDataValue>
        </targetBand>
        <targetBand>
          <name>Sigma0_HV</name>
          <type>float32</type>
          <expression>(Sigma0_HH == 0.0) ? -9999.0 : Sigma0_HV_db</expression>
          <noDataValue>-9999.0</noDataValue>
        </targetBand>
        <targetBand>
          <name>incidenceAngleFromEllipsoid</name>
          <type>float32</type>
          <expression>(Sigma0_HH == 0.0) ? -9999.0 : incidenceAngleFromEllipsoid</expression>
          <noDataValue>-9999.0</noDataValue>
        </targetBand>
      </targetBands>
    </parameters>
  </node>
  
  <node id="Write">
    <operator>Write</operator>
    <sources><sourceProduct refid="BandMaths"/></sources>
    <parameters><file>{final_dim}</file><formatName>BEAM-DIMAP</formatName></parameters>
  </node>
</graph>"""

# =========================================================================
# Graph 2: GLCM 提取 (步骤 9) 
# 为了防止内存溢出，将 GLCM 单独作为一个图运行
# =========================================================================
GLCM_GRAPH_TEMPLATE = """<graph id="GlcmMergeGraph">
  <version>1.0</version>
  <node id="Read-Final">
    <operator>Read</operator>
    <parameters><file>{input_dim}</file></parameters>
  </node>
  
  <node id="GLCM">
    <operator>GLCM</operator>
    <sources><sourceProduct refid="Read-Final"/></sources>
    <parameters>
      <sourceBands>Sigma0_HH</sourceBands>
      <angleStr>ALL</angleStr>
      <quantizerStr>Probabilistic Quantizer</quantizerStr>
      <quantizationLevelsStr>64</quantizationLevelsStr>
      <windowSizeStr>11x11</windowSizeStr>
      <displacement>2</displacement>
      <outputContrast>true</outputContrast>
      <outputDissimilarity>true</outputDissimilarity>
      <outputHomogeneity>true</outputHomogeneity>
      
      <outputASM>false</outputASM>
      <outputEnergy>false</outputEnergy>
      <outputMAX>false</outputMAX>
      <outputEntropy>false</outputEntropy>
      <outputMean>false</outputMean>
      <outputVariance>false</outputVariance>
      <outputCorrelation>false</outputCorrelation>
    </parameters>
  </node>
  
  <node id="BandMerge">
    <operator>BandMerge</operator>
    <sources>
      <sourceProduct refid="Read-Final"/>
      <sourceProduct.1 refid="GLCM"/>
    </sources>
  </node>
  
  <node id="Write">
    <operator>Write</operator>
    <sources><sourceProduct refid="BandMerge"/></sources>
    <parameters><file>{output_dim}</file><formatName>BEAM-DIMAP</formatName></parameters>
  </node>
</graph>"""

def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"函数 {func.__name__} 耗时: {end - start:.6f} 秒")
        return result
    return wrapper

@timer_decorator
def run_gpt(xml_path):
    cmd = f'"{GPT_PATH}" -J-Xmx10G -c 5G -q 8 -x "{xml_path}"'
    print(f"\n[执行 SNAP 命令] {cmd}")
    output_lines = []
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in iter(process.stdout.readline, b''):
        decoded = line.decode('utf-8', errors='ignore').strip()
        output_lines.append(decoded)
        print(decoded)
    process.wait()
    if process.returncode != 0:
        tail = "\n".join(output_lines[-120:])
        raise RuntimeError(
            f"SNAP 执行失败: {xml_path}, 退出码: {process.returncode}\n"
            f"------ SNAP 输出尾部 ------\n{tail}"
        )


def process_one_zip(input_zip, progress):
    paths = build_paths(input_zip)
    base_name = paths['base_name']
    did_run_gpt = False

    # 若第4步任意 NC 目录中已存在对应 NC 文件，说明该场景已完整处理，直接跳过。
    found_nc_path = None
    for nc_dir in OUTPUT_NC_DIRS:
        candidate = os.path.join(nc_dir, f"{base_name}.nc").replace('\\', '/')
        if os.path.isfile(candidate):
            found_nc_path = candidate
            break
    if found_nc_path:
        print(f"\n================ 跳过（NC已存在）: {base_name} ================")
        update_progress(progress, base_name, 'completed', f'检测到 NC 文件已存在，跳过处理: {found_nc_path}')
        return did_run_gpt

    # 若最终结果已存在，直接标记完成并跳过。
    if beam_dim_ready(paths['glcm_dim']):
        print(f"\n================ 跳过已完成: {base_name} ================")
        update_progress(progress, base_name, 'completed', '检测到已有最终输出，已跳过。')
        return did_run_gpt

    print(f"\n================ 正在处理: {base_name} ================")

    try:
        if beam_dim_ready(paths['final_dim']):
            print("=== 第一阶段：检测到已存在中间产物，跳过步骤 1-8 ===")
            update_progress(progress, base_name, 'prep_done', '检测到 final.dim，跳过预处理。')
        else:
            print("=== 第一阶段：执行预处理与辐射校正 (步骤 1-8) ===")
            with open(paths['prep_xml'], 'w') as f:
                f.write(PREP_GRAPH_TEMPLATE.format(input_zip=input_zip, final_dim=paths['final_dim']))
            did_run_gpt = True
            run_gpt(paths['prep_xml'])
            update_progress(progress, base_name, 'prep_done', '预处理阶段已完成。')

        if beam_dim_ready(paths['glcm_dim']):
            print("\n=== 第二阶段：检测到已存在最终产物，跳过步骤 9 ===")
        else:
            print("\n=== 第二阶段：提取 GLCM 纹理特征 (步骤 9) ===")
            with open(paths['glcm_xml'], 'w') as f:
                f.write(GLCM_GRAPH_TEMPLATE.format(input_dim=paths['final_dim'], output_dim=paths['glcm_dim']))
            did_run_gpt = True
            run_gpt(paths['glcm_xml'])

        if KEEP_INTERMEDIATE:
            print("\nKEEP_INTERMEDIATE=True，已保留中间文件与 XML。")
        else:
            print("\n=== 清理中间文件 ===")
            cleanup_after_glcm(paths)

        update_progress(progress, base_name, 'completed', '处理完成。')

    except Exception as e:
        error_text = str(e)
        if is_unreadable_s1_zip_error(error_text):
            print("\n[异常处理] 检测到不可读取的 Sentinel-1 压缩包，删除该 ZIP 并跳过继续。")
            try:
                remove_path(input_zip)
                update_progress(progress, base_name, 'skipped_bad_zip_deleted', 'ZIP 读入失败，已自动删除并跳过。')
            except Exception as del_err:
                update_progress(
                    progress,
                    base_name,
                    'skipped_bad_zip_delete_failed',
                    f'ZIP 读入失败，尝试删除失败: {del_err}'
                )
                print(f"[异常处理] 删除坏 ZIP 失败，继续跳过。原因: {del_err}")
            return did_run_gpt

        update_progress(progress, base_name, 'failed', error_text)
        raise

    print(f"\n文件处理完毕：{base_name}\n最终包含 GLCM 的文件保存在: {paths['glcm_dim']}")
    return did_run_gpt

def main():
    progress = load_progress()
    input_zips = list_input_zips(INPUT_DIR)
    print(f"检测到 {len(input_zips)} 个 SAR 压缩包，开始批量处理。")
    
    for input_zip in input_zips:
        did_run_gpt = process_one_zip(input_zip, progress)

        if did_run_gpt:
            print("\n[系统清理] 正在释放内存，清理可能残留的 SNAP/Java 僵尸进程...")
            try:
                # 强制干掉后台可能挂起的 gpt.exe 进程 (Windows 环境下)
                subprocess.run(["taskkill", "/F", "/IM", "gpt.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(["taskkill", "/F", "/IM", "java.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

            print("[系统清理] 暂停 15 秒，等待 Windows 操作系统回收虚拟内存 (Pagefile)...")
            time.sleep(15) # 强制等待 15 秒
            # ======================================
        else:
            print("[系统清理] 当前场景为跳过状态，已跳过 15 秒等待。")

    print("\n所有 SAR 压缩包处理完成。")

if __name__ == "__main__":
    main()