#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fast smoke test for 方案 2 (asymmetric CORAL thresholds). No data load / no model.

Checks:
  1. CORALLoss with under_alpha runs forward + backward, finite, on random data.
  2. Loss is non-decreasing in alpha (alpha scales a non-negative term).
  3. The 4 configs parse and route under_alpha through get_loss correctly,
     and decode is the neutral 'argmax'.
"""
import torch
from mmcv import Config

import src.functions as F
from src.losses import CORALLoss
from src.loaders import get_variable_options

torch.manual_seed(0)
B, Km1, H, W = 2, 4, 8, 8                      # K = 5
p = torch.rand(B, Km1, H, W).clamp(0.02, 0.98)  # P(y>k)
y = torch.randint(0, 5, (B, H, W))

print("== 1) forward/backward + finiteness ==")
for a in (1.0, 2.0, 4.0, 8.0):
    loss_fn = CORALLoss(num_classes=5, cost_matrix=F.NAVIGATION_COST_MATRIX, under_alpha=a)
    pin = p.clone().requires_grad_(True)
    l = loss_fn(pin, y)
    l.backward()
    assert torch.isfinite(l), f"non-finite loss at alpha={a}"
    assert pin.grad is not None and torch.isfinite(pin.grad).all(), f"bad grad at alpha={a}"
    print(f"  alpha={a}: loss={l.item():.4f}  grad_ok")

print("== 2) loss non-decreasing in alpha ==")
vals = [CORALLoss(5, F.NAVIGATION_COST_MATRIX, under_alpha=a)(p, y).item() for a in (1, 2, 4, 8)]
assert all(vals[i] <= vals[i + 1] + 1e-6 for i in range(len(vals) - 1)), vals
print(f"  {[round(v,4) for v in vals]}  OK")

print("== 3) config parse + get_loss plumbing + neutral decode ==")
for a in (1, 2, 4, 8):
    cfg = Config.fromfile(f"configs/_base_/PA2_swin_CORAL_asym_a{a}.py")
    opt = get_variable_options(cfg.train_options)
    assert opt["inference_decision"] == "argmax", f"a{a} decode={opt['inference_decision']}"
    assert opt.get("sod_head") == "coral", f"a{a} sod_head={opt.get('sod_head')}"
    lc = dict(opt["chart_loss"]["SOD"])
    built = F.get_loss(lc["type"], chart="SOD", **lc)
    assert abs(built.under_alpha - float(a)) < 1e-9, (a, built.under_alpha)
    print(f"  a{a}: under_alpha={built.under_alpha}  decode={opt['inference_decision']}  head={opt.get('sod_head')}  OK")

print("\nALL SMOKE PASS")
