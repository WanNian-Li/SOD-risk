#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CVaR q-sweep with optional temperature scaling (inference-only, NO retraining).

Tests the hypothesis that risk-averse CVaR decoding only helps once the softmax
is calibrated.  Pipeline:

  Phase A (if --temperature auto): one inference pass over the val set, fit a
          single scalar temperature T by NLL minimisation (Guo et al. 2017),
          and report Expected Calibration Error (ECE) before/after.
  Phase B: a second inference pass; for every scene the SAME logits are decoded
          under both the raw (T=1) and calibrated (T) softmax, across several
          risk-aversion levels q, accumulating a global confusion matrix each.

q = 1.0 recovers expected-cost decoding; q -> 0 is risk-averse (minimax-like).
"""

import argparse
import json
import os.path as osp
from collections import OrderedDict

import torch
from mmcv import Config

import src.functions as F
from src.functions import get_model, slide_inference
from src.loaders import AI4ArcticChallengeTestDataset, get_variable_options


def parse_args():
    p = argparse.ArgumentParser(description="CVaR q-sweep with temperature scaling.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--val-list", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cost-variant", default="current")
    p.add_argument("--q-list", default="1.0,0.5,0.25,0.1,0.05")
    p.add_argument("--temperature", default="auto",
                   help="'auto' to fit T on val NLL, or a float, or '1.0' for none")
    p.add_argument("--max-samples", type=int, default=1_000_000,
                   help="subsample budget for T fitting / ECE")
    p.add_argument("--output-csv", default="./cvar_q_sweep.csv")
    return p.parse_args()


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        return ckpt
    raise ValueError("Unsupported checkpoint format.")


def fit_temperature(logits, labels, device, max_iter=200):
    """Fit scalar T minimising NLL(logits / T, labels).  logits [N,K], labels [N]."""
    logits, labels = logits.to(device), labels.to(device)
    logT = torch.zeros(1, device=device, requires_grad=True)          # T = exp(logT) > 0
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=max_iter)
    ce = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = ce(logits / logT.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(logT.exp().item())


def ece(logits, labels, T, n_bins=15):
    """Expected Calibration Error of softmax(logits / T)."""
    probs = torch.softmax(logits / T, dim=1)
    conf, pred = probs.max(dim=1)
    acc = (pred == labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    total = 0.0
    N = labels.numel()
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            total += (m.float().mean() * (acc[m].mean() - conf[m].mean()).abs()).item()
    return total


def metrics_from_confusion(cm, cost_matrix):
    K = cm.shape[0]
    N = cm.sum().clamp(min=1.0)
    idx = torch.arange(K)
    dist = (idx.view(-1, 1) - idx.view(1, -1)).abs().float()
    lower = (idx.view(1, -1) < idx.view(-1, 1)).float()              # pred < true
    mean_cost = (cm * cost_matrix).sum() / N
    nrs = 1.0 - mean_cost / cost_matrix.max().clamp(min=1e-6)
    ovr = (cm * (dist >= 2).float()).sum() / N
    ur = (cm * lower).sum() / N
    oa = cm.diag().sum() / N
    tp = cm.diag(); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    f1 = (2 * tp) / (2 * tp + fp + fn).clamp(min=1e-9)
    iou = tp / (tp + fp + fn).clamp(min=1e-9)
    support = cm.sum(1)
    return dict(NRS=nrs.item(), OVR=ovr.item(), UR=ur.item(),
                mean_F1=f1.mean().item() * 100,
                weighted_F1=((f1 * support).sum() / N).item() * 100,
                mIoU=iou.mean().item() * 100, OA=oa.item() * 100,
                per_class_F1=[round(x * 100, 2) for x in f1.tolist()])


def main():
    args = parse_args()
    q_list = [float(x) for x in args.q_list.split(",")]

    cfg = Config.fromfile(args.config)
    train_options = get_variable_options(cfg.train_options)
    train_options["path_to_train_data"] = args.data_root
    train_options["path_to_test_data"] = args.data_root

    F.select_cost_matrix(args.cost_variant)
    device = torch.device(args.device)
    cost_cpu = F.NAVIGATION_COST_MATRIX.float().cpu()

    net = get_model(train_options, device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = extract_state_dict(ckpt)
    sd = OrderedDict((k.replace("module.", "").replace("_orig_mod.", ""), v) for k, v in sd.items())
    net.load_state_dict(sd, strict=True)
    net.eval()

    with open(args.val_list, "r", encoding="utf-8") as f:
        val_scenes = json.load(f)
    dataset = AI4ArcticChallengeTestDataset(options=train_options, files=val_scenes, mode="train")

    def loader():
        return torch.utils.data.DataLoader(dataset, batch_size=None,
                                           num_workers=train_options.get("num_workers_val", 4),
                                           shuffle=False)

    chart = train_options.get("target_chart", "SOD")
    K = train_options["n_classes"][chart]
    print(f"Scenes: {len(val_scenes)} | chart: {chart} | K: {K} | "
          f"cost: {args.cost_variant} (C_max={cost_cpu.max():.0f}) | q: {q_list}")

    def infer(inf_x):
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            out = slide_inference(inf_x.to(device, non_blocking=True), net, train_options, "val")
        return out[chart].detach().float()                            # [1, K, h, w] on GPU

    # ---- Phase A: fit temperature on val NLL (logits at model resolution) ----
    if args.temperature == "auto":
        buf_logits, buf_labels, collected = [], [], 0
        per_scene = max(1, args.max_samples // max(1, len(val_scenes)))
        for inf_x, inf_y, cfv_masks, _, scene_name, original_size in loader():
            logits = infer(inf_x)
            h, w = logits.shape[-2:]
            true_ds = torch.nn.functional.interpolate(
                inf_y[chart][None, None].float(), size=(h, w), mode="nearest").long().squeeze().to(device)
            mask_ds = torch.nn.functional.interpolate(
                cfv_masks[chart][None, None].to(torch.uint8), size=(h, w), mode="nearest").squeeze().to(device)
            valid = (~torch.gt(mask_ds, 0)) & (true_ds >= 0) & (true_ds < K)
            lf = logits[0].permute(1, 2, 0)[valid]                    # [n, K]
            yf = true_ds[valid]                                       # [n]
            if lf.shape[0] > per_scene:
                sel = torch.randperm(lf.shape[0], device=device)[:per_scene]
                lf, yf = lf[sel], yf[sel]
            buf_logits.append(lf.cpu()); buf_labels.append(yf.cpu()); collected += lf.shape[0]
            del logits, inf_x
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        L = torch.cat(buf_logits); Y = torch.cat(buf_labels)
        T = fit_temperature(L, Y, device)
        e0, e1 = ece(L, Y, 1.0), ece(L, Y, T)
        print(f"\n[temperature] fitted T = {T:.4f} on {collected:,} pixels | "
              f"ECE: {e0:.4f} (T=1) -> {e1:.4f} (T={T:.3f})\n")
    else:
        T = float(args.temperature)
        print(f"\n[temperature] using fixed T = {T}\n")

    # ---- Phase B: q-sweep for raw (T=1) and calibrated (T) on identical logits ----
    cm_raw = {q: torch.zeros(K, K, dtype=torch.float64) for q in q_list}
    cm_cal = {q: torch.zeros(K, K, dtype=torch.float64) for q in q_list}

    def accumulate(cm_dict, probs, true_flat, valid):
        for q in q_list:
            pred = F.cvar_optimal_decision(probs, F.NAVIGATION_COST_MATRIX, q=q).squeeze(0).long()
            pf = pred[valid]
            sel = (true_flat >= 0) & (true_flat < K) & (pf >= 0) & (pf < K)
            binc = torch.bincount(true_flat[sel] * K + pf[sel], minlength=K * K).reshape(K, K).double().cpu()
            cm_dict[q] += binc

    for inf_x, inf_y, cfv_masks, _, scene_name, original_size in loader():
        logits = infer(inf_x)
        del inf_x
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        masks_int = torch.nn.functional.interpolate(
            cfv_masks[chart][None, None].to(torch.uint8), size=original_size, mode="nearest").squeeze().squeeze()
        valid = (~torch.gt(masks_int, 0)).to(device)
        if train_options.get("down_sample_scale", 1) != 1:
            logits = torch.nn.functional.interpolate(logits, size=original_size, mode="nearest")
        true = torch.nn.functional.interpolate(
            inf_y[chart][None, None].float(), size=original_size, mode="nearest").long().squeeze().to(device)
        true_flat = true[valid]

        probs_raw = torch.softmax(logits, dim=1)
        accumulate(cm_raw, probs_raw, true_flat, valid)
        del probs_raw
        probs_cal = torch.softmax(logits / T, dim=1)
        accumulate(cm_cal, probs_cal, true_flat, valid)
        del probs_cal, logits

    def print_table(title, cm):
        print(f"\n================ {title} ================")
        hdr = f"{'q':>6} | {'NRS':>7} | {'OVR':>7} | {'UR':>7} | {'meanF1':>7} | {'wF1':>6} | {'mIoU':>6} | {'OA':>6}"
        print(hdr); print("-" * len(hdr))
        rows = []
        for q in q_list:
            m = metrics_from_confusion(cm[q], cost_cpu); m["q"] = q
            print(f"{q:>6} | {m['NRS']:.4f}  | {m['OVR']:.4f}  | {m['UR']:.4f}  | "
                  f"{m['mean_F1']:6.2f}  | {m['weighted_F1']:5.2f} | {m['mIoU']:5.2f} | {m['OA']:5.2f}")
            print(f"        per-class F1 (W/NI/TFYI/ThkFYI/MYI): {m['per_class_F1']}")
            rows.append(m)
        return rows

    rows_raw = print_table(f"RAW  softmax (T=1)", cm_raw)
    rows_cal = print_table(f"CALIBRATED softmax (T={T:.4f})", cm_cal)

    import csv
    keys = ["variant", "q", "NRS", "OVR", "UR", "mean_F1", "weighted_F1", "mIoU", "OA", "per_class_F1"]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows_raw:
            w.writerow({"variant": "raw", **{k: r[k] for k in keys[1:]}})
        for r in rows_cal:
            w.writerow({"variant": f"cal_T{T:.3f}", **{k: r[k] for k in keys[1:]}})
    print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    main()
