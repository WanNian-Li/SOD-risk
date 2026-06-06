import os
import re
import time
import warnings
from datetime import datetime, timedelta

import asf_search as asf

warnings.filterwarnings("ignore")

os.environ["http_proxy"] = "http://127.0.0.1:7897"
os.environ["https_proxy"] = "http://127.0.0.1:7897"

# 第4步 NC 文件可能存在的多个输出目录（任意一个中找到 {base_name}.nc 即跳过下载该场景）
OUTPUT_NC_DIRS = [
    r"F:/ZJU/11_Ice/dataset_create/data_nc_CAA",    # 目录1
    r"F:/ZJU/11_Ice/dataset_create/data_nc_24_2-4", # 目录2
    r"F:/ZJU/11_Ice/dataset_create/data_nc_cls34",  # 目录3
    r"F:/ZJU/11_Ice/dataset_create/data_nc_hb",     # 目录4
    r"F:/ZJU/11_Ice/dataset_create/data_nc_wa",     # 目录5
    r"F:/ZJU/11_Ice/dataset_create/dataset_nc_new", # 目录6
]


def nc_already_exists(scene_name):
    """检查任意 NC 目录中是否已存在该场景对应的 NC 文件。"""
    base_name = os.path.splitext(scene_name)[0]
    for nc_dir in OUTPUT_NC_DIRS:
        candidate = os.path.join(nc_dir, f"{base_name}.nc").replace('\\', '/')
        if os.path.isfile(candidate):
            return True
    return False


def extract_ice_dates(cis_root_dir):
    if not os.path.isdir(cis_root_dir):
        raise FileNotFoundError(f"CIS 目录不存在: {cis_root_dir}")

    date_pattern = re.compile(r"_(\d{8})_")
    date_set = set()

    for name in os.listdir(cis_root_dir):
        folder_path = os.path.join(cis_root_dir, name)
        if not os.path.isdir(folder_path):
            continue

        match = date_pattern.search(name)
        if not match:
            continue

        try:
            date_obj = datetime.strptime(match.group(1), "%Y%m%d").date()
            date_set.add(date_obj)
        except ValueError:
            continue

    return sorted(date_set)


def build_scene_index_by_date(search_results):
    scenes_by_date = {}
    for scene in search_results:
        start_time = scene.properties.get("startTime", "")
        if "T" not in start_time:
            continue
        date_str = start_time.split("T")[0]
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        scenes_by_date.setdefault(date_obj, []).append(scene)
    return scenes_by_date


def choose_scenes_for_ice_date(ice_date, scenes_by_date):
    candidate_dates = [
        ice_date,
        ice_date - timedelta(days=1),
        ice_date + timedelta(days=1),
    ]

    for candidate_date in candidate_dates:
        scene_list = scenes_by_date.get(candidate_date, [])
        if scene_list:
            # Return all scenes for the first matched candidate date.
            return scene_list, candidate_date

    return [], None


def write_report(report_path, matched_records, unmatched_dates):
    matched_ice_dates = sorted({record["ice_date"] for record in matched_records})
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("Sentinel-1 与 CIS 冰图日期匹配报告\n")
        file.write(f"匹配成功冰图日期数量: {len(matched_ice_dates)}\n")
        file.write(f"匹配到的 SAR 场景数量: {len(matched_records)}\n")
        file.write(f"匹配失败冰图日期数量: {len(unmatched_dates)}\n\n")

        file.write("[成功匹配日期]\n")
        for record in matched_records:
            file.write(
                f"冰图日期: {record['ice_date']} -> SAR日期: {record['sar_date']} -> 场景: {record['scene_name']}\n"
            )

        file.write("\n[未匹配日期]\n")
        for date_obj in unmatched_dates:
            file.write(f"{date_obj.isoformat()}\n")


def download_s1_custom_dataset():
    print("=== Sentinel-1 自动化检索与下载启动 ===")

    username = "clover_lwn"
    password = "Lwn20030913."

    # cis_root_dir = r"F:\ZJU\11_Ice\dataset_create\Icechart\CIS"
    cis_root_dir = r"F:\ZJU\11_Ice\dataset_create\Icechart\CIS_CAA"
    download_dir = r"F:\ZJU\11_Ice\dataset_create\S1_raw_CAA"
    os.makedirs(download_dir, exist_ok=True)

    # aoi_wkt = "POLYGON((-90 60,-80 60,-80 62,-90 62,-90 60))" # Hudson Bay 1
    # aoi_wkt = "POLYGON((-85.8527 59.9403,-81.8054 59.9424,-82.3897 60.9192,-85.0055 60.9192,-85.8527 59.9403))" # Hudson Bay 2
    # aoi_wkt = "POLYGON((-136.7218 73.31,-133.6691 72.8488,-129.9707 74.2444,-127.4331 75.5772,-132.2015 75.4703,-134.6671 74.5444,-136.7218 73.31))"
    # aoi_wkt = "POLYGON((-127.5755 75.093,-126.9586 75.093,-126.9586 75.3225,-127.5755 75.3225,-127.5755 75.093))"
    aoi_wkt = "POINT(-107.7537 74.3821)" # CAA加拿大北极群岛内部水道

    print("\n[1/5] 读取 CIS 文件夹并提取日期...")
    try:
        ice_dates = extract_ice_dates(cis_root_dir)
    except FileNotFoundError as error:
        print(error)
        return

    if not ice_dates:
        print("未在 CIS 子目录名中提取到有效日期（格式应包含 _YYYYMMDD_）。")
        return

    print(f"      -> 共提取到 {len(ice_dates)} 个冰图日期。")

    search_start = (ice_dates[0] - timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
    search_end = (ice_dates[-1] + timedelta(days=1)).strftime("%Y-%m-%dT23:59:59Z")

    print("\n[2/5] 在 ASF 按时间范围检索 SAR 影像...")
    print(f"      - 时间: {search_start} 至 {search_end}")

    results = asf.geo_search(
        platform=[asf.PLATFORM.SENTINEL1],
        processingLevel=asf.PRODUCT_TYPE.GRD_MD,
        beamMode=asf.BEAMMODE.EW,
        start=search_start,
        end=search_end,
        intersectsWith=aoi_wkt,
    )

    print(f"      -> ASF 检索完成，共返回 {len(results)} 景影像。")
    if len(results) == 0:
        print("检索结果为空，无法进行日期匹配。")
        return

    scenes_by_date = build_scene_index_by_date(results)

    print("\n[3/5] 进行日期匹配（当日 -> 前一天 -> 后一天），每个日期可匹配多景...")
    matched_records = []
    unmatched_dates = []
    selected_scene_ids = set()
    nc_skipped_count = 0

    for ice_date in ice_dates:
        chosen_scenes, sar_date = choose_scenes_for_ice_date(ice_date, scenes_by_date)
        if not chosen_scenes:
            unmatched_dates.append(ice_date)
            continue

        for scene in chosen_scenes:
            scene_name = scene.properties.get("fileName", "UNKNOWN")
            scene_id = scene.properties.get("sceneName", scene_name)

            if nc_already_exists(scene_name):
                nc_skipped_count += 1
                print(f"      [跳过] NC 已存在，无需下载: {scene_name}")
                continue

            selected_scene_ids.add(scene_id)
            matched_records.append(
                {
                    "ice_date": ice_date.isoformat(),
                    "sar_date": sar_date.isoformat(),
                    "scene_name": scene_name,
                    "scene_id": scene_id,
                }
            )

    matched_ice_date_count = len({record["ice_date"] for record in matched_records})
    print(f"      -> 匹配成功冰图日期: {matched_ice_date_count}")
    print(f"      -> 匹配到的 SAR 场景: {len(matched_records)}")
    print(f"      -> NC 已存在跳过下载: {nc_skipped_count}")
    print(f"      -> 匹配失败冰图日期: {len(unmatched_dates)}")

    print("\n[4/5] 生成匹配报告...")
    report_path = os.path.join(download_dir, "match_report.txt")
    write_report(report_path, matched_records, unmatched_dates)
    print(f"      -> 报告已保存: {report_path}")

    input("\n按回车键开始下载匹配到的 SAR 影像...")  # 等待用户确认后再开始下载
    if not selected_scene_ids:
        print("没有可下载的 SAR 影像，流程结束。")
        return

    scene_id_list = sorted(selected_scene_ids)
    final_results = asf.granule_search(scene_id_list)

    print("\n[5/5] 验证账号并下载匹配到的 SAR 压缩包...")
    try:
        session = asf.ASFSession().auth_with_creds(username, password)
    except Exception as error:
        print(f"账户验证失败，请检查账号密码。错误信息: {error}")
        return
    # input("按回车键开始下载...")
    print("正在下载匹配到的 SAR 影像，请耐心等待...")
    max_retries = 50
    retry_count = 0

    while retry_count < max_retries:
        try:
            final_results.download(path=download_dir, session=session, processes=3)
            print("\n=== 下载完成 ===")
            break
        except Exception as error:
            retry_count += 1
            print(f"\n[!] 下载中断: {error}")
            print(f"    30 秒后进行第 {retry_count}/{max_retries} 次重试...")
            time.sleep(30)

    if retry_count >= max_retries:
        print("\n[!] 达到最大重试次数，下载终止。")

    print("\n=== 匹配统计汇总 ===")
    print(f"获取到的 SAR 图像数量: {len(matched_records)}")
    print(f"获取到的冰图日期数量: {matched_ice_date_count}")
    print(f"未获取到的冰图日期数量: {len(unmatched_dates)}")
    if unmatched_dates:
        print("未获取日期:")
        for date_obj in unmatched_dates:
            print(f"- {date_obj.isoformat()}")


if __name__ == "__main__":
    download_s1_custom_dataset()
