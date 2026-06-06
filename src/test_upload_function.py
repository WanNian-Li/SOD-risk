
# # AutoICE - test model and prepare upload package
# This notebook tests the 'best_model', created in the quickstart notebook,
# with the tests scenes exempt of reference data.
# The model outputs are stored per scene and chart in an xarray Dataset in individual Dataarrays.
# The xarray Dataset is saved and compressed in an .nc file ready to be uploaded to the AI4EO.eu platform.
# Finally, the scene chart inference is shown.
#
# The first cell imports necessary packages:

# -- Built-in modules -- #
import json
import os
import os.path as osp

# -- Third-part modules -- #
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from tqdm import tqdm
from mmengine.utils import mkdir_or_exist
import wandb
# --Proprietary modules -- #
from src.functions import chart_cbar, water_edge_plot_overlay, compute_metrics, water_edge_metric, class_decider
from src.loaders import AI4ArcticChallengeTestDataset, get_variable_options
from src.functions import slide_inference, batched_slide_inference



def test(mode: str, net: torch.nn.modules, checkpoint: str, device: str, cfg, test_list, test_name, test_list_reference=None):
    """_summary_

    Args:
        net (torch.nn.modules): The model
        checkpoint (str): The checkpoint to the model
        device (str): The device to run the inference on
        cfg (Config): mmengine based Config object, Can be considered dict
    """

    if mode not in ["val", "test", "apply"]:
        raise ValueError("mode must be one of 'val', 'test', 'apply'")

    # 'apply' 模式：无任何标签的真实推理（配合 4b_pack_for_inference.py 的 NC 使用）
    # - dataset 内部用 mode='test_no_gt'，不读取 SIC/SOD/FLOE
    # - 本函数内跳过所有依赖标签的逻辑（指标计算、混淆矩阵、GT 可视化、wandb summary 等）
    _apply_mode = (mode == 'apply')

    train_options = cfg.train_options
    train_options = get_variable_options(train_options)
    weights = torch.load(checkpoint, weights_only=False)['model_state_dict']
    # 兼容 torch.compile 训练的 checkpoint：剥掉 "_orig_mod." 前缀
    if any(k.startswith('_orig_mod.') for k in weights.keys()):
        weights = {k.replace('_orig_mod.', '', 1): v for k, v in weights.items()}

    # Setup U-Net model, adam optimizer, loss function and dataloader.
    # net = UNet(options=train_options).to(device)
    net.load_state_dict(weights)
    print('Model successfully loaded.')
    experiment_name = osp.splitext(osp.basename(cfg.work_dir))[0]
    artifact = wandb.Artifact(experiment_name, 'dataset')
    table = wandb.Table(columns=['ID', 'Image'])

    # - Stores the output and the reference pixels to calculate the scores after inference on all the scenes.
    output_class = {chart: torch.Tensor().to(device) for chart in train_options['charts']}
    outputs_flat = {chart: torch.Tensor().to(device) for chart in train_options['charts']}
    inf_ys_flat = {chart: torch.Tensor().to(device) for chart in train_options['charts']}
    # Outputs mask by train fill values
    outputs_tfv_mask = {chart: torch.Tensor().to(device) for chart in train_options['charts']}




    # ### Prepare the scene list, dataset and dataloaders

    if mode == 'test':

        train_options['test_list'] = test_list
        # The test data is stored in a separate folder inside the training data.
        # upload_package = xr.Dataset()  # To store model outputs.
        dataset = AI4ArcticChallengeTestDataset(
            options=train_options, files=train_options['test_list'], mode='test')
        asid_loader = torch.utils.data.DataLoader(
            dataset, batch_size=None, num_workers=train_options['num_workers_val'], shuffle=False)
        print('Setup ready')

    elif mode == 'val':
        train_options['test_list'] = test_list
        # The test data is stored in a separate folder inside the training data.
        # upload_package = xr.Dataset()  # To store model outputs.
        dataset = AI4ArcticChallengeTestDataset(
            options=train_options, files=train_options['test_list'], mode='train')
        asid_loader = torch.utils.data.DataLoader(
            dataset, batch_size=None, num_workers=train_options['num_workers_val'], shuffle=False)
        print('Setup ready')

    elif mode == 'apply':
        train_options['test_list'] = test_list
        dataset = AI4ArcticChallengeTestDataset(
            options=train_options, files=train_options['test_list'], mode='test_no_gt')
        asid_loader = torch.utils.data.DataLoader(
            dataset, batch_size=None, num_workers=train_options['num_workers_val'], shuffle=False)
        print('Setup ready (apply / no-GT)')

    if mode == 'val':
        inference_name = 'inference_val'
    elif mode == 'test':
        inference_name = 'inference_test'
    elif mode == 'apply':
        inference_name = 'inference_apply'

    # 推理/验证数据根目录（用于回读 coastline_land_mask 做陆地可视化）
    if mode in ('test', 'apply'):
        _scene_root = train_options.get('path_to_test_data', None)
    else:
        _scene_root = train_options.get('path_to_train_data', None)

    # 构建陆地专用 colormap：正常类别走 jet，陆地（<vmin）→ 浅灰，NaN → 透明
    _land_cmap = plt.get_cmap('jet').copy()
    _land_cmap.set_under('lightgray')
    _land_cmap.set_bad(color='white', alpha=0.0)

    os.makedirs(osp.join(cfg.work_dir, inference_name), exist_ok=True)
    net.eval()
    for inf_x, inf_y, cfv_masks, tfv_mask, scene_name, original_size in tqdm(iterable=asid_loader,
                                                               total=len(train_options['test_list']), colour='green', position=0):
        scene_name = osp.splitext(scene_name)[0]  # Remove the .nc extension.
        torch.cuda.empty_cache()

        # 读取该场景的海岸线陆地掩膜（若 NC 中存在）
        coastline_mask_full = None
        if _scene_root is not None:
            _nc_path = osp.join(_scene_root, scene_name + '.nc')
            try:
                with xr.open_dataset(_nc_path, engine='h5netcdf') as _ds_land:
                    if 'coastline_land_mask' in _ds_land.variables:
                        coastline_mask_full = _ds_land['coastline_land_mask'].values.astype(bool)
            except Exception as _e:
                # 老 NC 没有此变量时静默跳过，不影响推理流程
                coastline_mask_full = None

        inf_x = inf_x.to(device, non_blocking=True)
        with torch.no_grad(), torch.cuda.amp.autocast():
            _needs_slide = (train_options['model_selection'] == 'swin' or
                            inf_x.shape[2] > train_options['patch_size'] or
                            inf_x.shape[3] > train_options['patch_size'])
            if _needs_slide:
                output = slide_inference(inf_x, net, train_options, 'test')
                # output = batched_slide_inference(inf_x, net, train_options, 'test')
            else:
                output = net(inf_x)

            # output storage as a flat tensor
            # if test is False:
            # for chart in train_options['charts']:
            
            # if test:
            #     masks_int = masks.to(torch.uint8)
            #     masks_int = torch.nn.functional.interpolate(masks_int.unsqueeze(
            #         0).unsqueeze(0), size=original_size, mode='nearest').squeeze().squeeze()
            #     masks = torch.gt(masks_int, 0)
            #     tfv_mask = (inf_x.squeeze()[0, :, :] == train_options['train_fill_value']).squeeze()
            #     tfv_mask = torch.nn.functional.interpolate(tfv_mask.type(torch.uint8).unsqueeze(
            #         0).unsqueeze(0), size=original_size, mode='nearest').squeeze().squeeze().to(torch.bool)
            # else:

            # Up sample the masks
            tfv_mask = torch.nn.functional.interpolate(tfv_mask.type(torch.uint8).unsqueeze(0).unsqueeze(0), size=original_size, mode='nearest').squeeze().squeeze().to(torch.bool)
            if cfv_masks is not None:
                for chart in train_options['charts']:
                    masks_int = cfv_masks[chart].to(torch.uint8)
                    masks_int = torch.nn.functional.interpolate(masks_int.unsqueeze(
                        0).unsqueeze(0), size=original_size, mode='nearest').squeeze().squeeze()
                    cfv_masks[chart] = torch.gt(masks_int, 0)

            # Upsample data
            if train_options['down_sample_scale'] != 1:
                for chart in train_options['charts']:
                    # check if the output is regression output, if yes, permute the dimension
                    if output[chart].size(3) == 1:
                        output[chart] = output[chart].permute(0, 3, 1, 2)
                        output[chart] = torch.nn.functional.interpolate(
                            output[chart], size=original_size, mode='nearest')
                        output[chart] = output[chart].permute(0, 2, 3, 1)
                    else:
                        # 先 argmax 降维（5ch→1ch），再上采样，节省 ~5x 显存
                        output[chart] = class_decider(output[chart], train_options, chart).unsqueeze(0).unsqueeze(0).float()
                        output[chart] = torch.nn.functional.interpolate(
                            output[chart], size=original_size, mode='nearest').squeeze().to(torch.int64)

                    # upscale the output
                    # if not test:
                    if inf_y is not None:
                        inf_y[chart] = torch.nn.functional.interpolate(inf_y[chart].unsqueeze(dim=0).unsqueeze(dim=0),
                                                                       size=original_size, mode='nearest').squeeze()

        # for chart in train_options['charts']:
        #     # check if the output is regression output, if yes, round the output to integer
        #     # TODO class decider function in here
        #     output[chart] = class_decider(output[chart], train_options, chart)
        #     output[chart] = output[chart].cpu().numpy()
        #     # if test:
        #     #     upload_package[f"{scene_name}_{chart}"] = xr.DataArray(name=f"{scene_name}_{chart}", data=output[chart].astype('uint8'),
        #     #                                                            dims=(f"{scene_name}_{chart}_dim0", f"{scene_name}_{chart}_dim1"))
        #     # else:
        #     inf_y[chart] = inf_y[chart].squeeze().cpu().numpy()

        # output storage as a flat tensor
        # if test is False:
            # for chart in train_options['charts']:
            #     outputs_flat[chart] = torch.cat(
            #         (outputs_flat[chart], torch.tensor(output[chart][~masks[chart]]).to(device)))
            #     outputs_tfv_mask[chart] = torch.cat(
            #         (outputs_tfv_mask[chart], torch.tensor(output[chart])[~tfv_mask].to(device)))
            #     inf_ys_flat[chart] = torch.cat((inf_ys_flat[chart], torch.tensor(inf_y[chart]
            #                                     [~masks[chart]]).to(device, non_blocking=True)))
        for chart in train_options['charts']:
            # down_sample_scale != 1 の場合は upsample ブロック内で既に argmax 済み
            if train_options['down_sample_scale'] != 1 and output[chart].dim() == 2:
                output_class[chart] = output[chart].detach()
            else:
                output_class[chart] = class_decider(output[chart], train_options, chart).detach()
            # apply 模式没有标签，跳过 flat 张量累积（后续指标/混淆矩阵也会跳过）
            if not _apply_mode:
                outputs_flat[chart] = torch.cat(
                            (outputs_flat[chart], output_class[chart][~cfv_masks[chart]]))
                outputs_tfv_mask[chart] = torch.cat(
                            (outputs_tfv_mask[chart], output_class[chart][~tfv_mask].to(device)))
                inf_ys_flat[chart] = torch.cat(
                            (inf_ys_flat[chart], inf_y[chart][~cfv_masks[chart]].to(device, non_blocking=True)))

        for chart in train_options['charts']:
            if inf_y is not None:
                inf_y[chart] = inf_y[chart].cpu().numpy()
            output_class[chart] = output_class[chart].squeeze().cpu().numpy()

        # - Show the scene inference.
        # Layout: 2×3
        #   Row 0: HH | HV | GLCM Contrast
        #   Row 1: GLCM Homogeneity | SOD Prediction | SOD Ground Truth
        fig, axs2d = plt.subplots(nrows=2, ncols=3, figsize=(18, 11))
        axs = axs2d.flat

        img_data = torch.squeeze(inf_x, dim=0).cpu().numpy()

        def pclip(arr, lo=2, hi=98):
            """Percentile-based vmin/vmax to suppress outliers."""
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return arr.min(), arr.max()
            return np.percentile(finite, lo), np.percentile(finite, hi)

        # --- Row 0, Col 0: HH ---
        ax = axs[0]
        hh = img_data[0]
        vmin, vmax = pclip(hh)
        ax.imshow(hh, cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title('SAR HH', fontsize=12, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

        # --- Row 0, Col 1: HV ---
        ax = axs[1]
        hv = img_data[1]
        vmin, vmax = pclip(hv)
        ax.imshow(hv, cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title('SAR HV', fontsize=12, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

        # --- Row 0, Col 2: GLCM Contrast ---
        ax = axs[2]
        glcm_contrast = img_data[3]   # index 3: glcm_sigma0_hh_contrast
        vmin, vmax = pclip(glcm_contrast)
        im = ax.imshow(glcm_contrast, cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title('GLCM Contrast (HH)', fontsize=12, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # --- Row 1, Col 0: GLCM Homogeneity ---
        ax = axs[3]
        glcm_homogeneity = img_data[5]   # index 5: glcm_sigma0_hh_homogeneity
        vmin, vmax = pclip(glcm_homogeneity)
        im = ax.imshow(glcm_homogeneity, cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title('GLCM Homogeneity (HH)', fontsize=12, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # --- Row 1, Col 1: SOD Prediction ---
        ax = axs[4]
        sod_pred = output_class['SOD'].astype(float)
        # apply 模式用 tfv_mask 作为 invalid 兜底（cfv_masks 为 None）
        _invalid_pred = cfv_masks['SOD'].numpy() if (cfv_masks is not None) else tfv_mask.cpu().numpy()
        # 先把无效/fill 区域统一置 NaN（透明显示）
        sod_pred[_invalid_pred] = np.nan
        # 再把陆地置为 -1（< vmin），由 cmap.set_under('lightgray') 渲染为灰色
        if coastline_mask_full is not None:
            sod_pred[coastline_mask_full] = -1.0
        ax.imshow(sod_pred, vmin=0, vmax=train_options['n_classes']['SOD'] - 2,
                  cmap=_land_cmap, interpolation='nearest')
        ax.set_title('SOD: Prediction', fontsize=12, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        chart_cbar(ax=ax, n_classes=train_options['n_classes']['SOD'], chart='SOD', cmap='jet')

        # --- Row 1, Col 2 ---
        # val/test：SOD Ground Truth；apply：HH + 陆地叠加（用于视觉核验掩膜对齐）
        ax = axs[5]
        if _apply_mode or inf_y is None:
            hh_gray = img_data[0].copy()
            vmin_hh, vmax_hh = pclip(hh_gray)
            ax.imshow(hh_gray, cmap='gray', vmin=vmin_hh, vmax=vmax_hh, interpolation='nearest')
            if coastline_mask_full is not None:
                overlay = np.zeros((*coastline_mask_full.shape, 4), dtype=np.float32)
                overlay[coastline_mask_full] = [1.0, 0.3, 0.3, 0.45]  # 半透明红
                ax.imshow(overlay, interpolation='nearest')
            ax.set_title('HH + Land Overlay (apply mode)', fontsize=12, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])
        else:
            sod_gt = inf_y['SOD'].astype(float)
            sod_gt[cfv_masks['SOD'].numpy()] = np.nan
            if coastline_mask_full is not None:
                sod_gt[coastline_mask_full] = -1.0
            ax.imshow(sod_gt, vmin=0, vmax=train_options['n_classes']['SOD'] - 2,
                      cmap=_land_cmap, interpolation='nearest')
            ax.set_title('SOD: Ground Truth', fontsize=12, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])
            chart_cbar(ax=ax, n_classes=train_options['n_classes']['SOD'], chart='SOD', cmap='jet')

        fig.suptitle(f'Scene: {scene_name}', fontsize=13, y=1.01)
        plt.tight_layout()
        fig.savefig(f"{osp.join(cfg.work_dir,inference_name,scene_name)}.png",
                    format='png', dpi=128, bbox_inches="tight")
        plt.close('all')
        table.add_data(scene_name, wandb.Image(f"{osp.join(cfg.work_dir,inference_name,scene_name)}.png"))


    # apply 模式没有 GT，跳过所有指标 / 混淆矩阵 / wandb summary，直接收尾
    if _apply_mode:
        artifact.add(table, experiment_name + '_apply')
        wandb.log_artifact(artifact)
        return

    # compute combine score
    combined_score, scores = compute_metrics(true=inf_ys_flat, pred=outputs_flat, charts=train_options['charts'],
                                             metrics=train_options['chart_metric'], num_classes=train_options['n_classes'])

    # compute water edge metric
    water_edge_accuarcy = water_edge_metric(outputs_tfv_mask, train_options)
    if train_options['compute_classwise_f1score']:
        from functions import compute_classwise_f1score
        classwise_scores = compute_classwise_f1score(true=inf_ys_flat, pred=outputs_flat,
                                                     charts=train_options['charts'], num_classes=train_options['n_classes'])

    if train_options['plot_confusion_matrix']:
        from torchmetrics.functional.classification import multiclass_confusion_matrix
        import seaborn as sns
        from utils import GROUP_NAMES

        for chart in train_options['charts']:
            cm = multiclass_confusion_matrix(
                preds=outputs_flat[chart], target=inf_ys_flat[chart], num_classes=train_options['n_classes'][chart])
            # Calculate percentages
            cm = cm.cpu().numpy()
            cm_percent = np.round(cm / cm.sum(axis=1)[:, np.newaxis] * 100, 2)
            # Plot the confusion matrix
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(cm_percent, annot=True, cmap='Blues')
            # Customize the plot
            class_names = list(GROUP_NAMES[chart].values())
            class_names.append('255')
            tick_marks = np.arange(len(class_names)) + 0.5
            plt.xticks(tick_marks, class_names, rotation=45)
            if chart in ['FLOE', 'SOD']:
                plt.yticks(tick_marks, class_names, rotation=45)
            else:
                plt.yticks(tick_marks, class_names)

            plt.xlabel('Predicted Labels')
            plt.ylabel('Actual Labels')
            plt.title('Confusion Matrix')
            cbar = ax.collections[0].colorbar
            # cbar.set_ticks([0, .2, .75, 1])
            cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
            mkdir_or_exist(f"{osp.join(cfg.work_dir)}/{test_name}")
            plt.savefig(f"{osp.join(cfg.work_dir)}/{test_name}/{chart}_confusion_matrix.png",
                        format='png', dpi=128, bbox_inches="tight")

    wandb.run.summary[f"{test_name}/Best Combined Score"] = combined_score
    print(f"{test_name}/Best Combined Score = {combined_score}")
    for chart in train_options['charts']:
        wandb.run.summary[f"{test_name}/{chart} {train_options['chart_metric'][chart]['func'].__name__}"] = scores[chart]
        print(
            f"{test_name}/{chart} {train_options['chart_metric'][chart]['func'].__name__} = {scores[chart]}")
        if train_options['compute_classwise_f1score']:
            wandb.run.summary[f"{test_name}/{chart}: classwise score:"] = classwise_scores[chart]
            print(
                f"{test_name}/{chart}: classwise score: = {classwise_scores[chart]}")

    wandb.run.summary[f"{test_name}/Water Consistency Accuarcy"] = water_edge_accuarcy
    print(
        f"{test_name}/Water Consistency Accuarcy = {water_edge_accuarcy}")

    if mode == 'test':
        artifact.add(table, experiment_name+'_test')
    elif mode == 'val':
        artifact.add(table, experiment_name+'_val')
    
    wandb.log_artifact(artifact)

    # # - Save upload_package with zlib compression.
    # if test:
    #     print('Saving upload_package. Compressing data with zlib.')
    #     compression = dict(zlib=True, complevel=1)
    #     encoding = {var: compression for var in upload_package.data_vars}
    #     upload_package.to_netcdf(osp.join(cfg.work_dir, f'{experiment_name}_upload_package.nc'),
    #                              # f'{osp.splitext(osp.basename(cfg))[0]}
    #                              mode='w', format='netcdf4', engine='h5netcdf', encoding=encoding)
    #     print('Testing completed.')
    #     print("File saved succesfully at", osp.join(cfg.work_dir, f'{experiment_name}_upload_package.nc'))
    #     wandb.save(osp.join(cfg.work_dir, f'{experiment_name}_upload_package.nc'))
