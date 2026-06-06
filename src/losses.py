__author__ = 'Muhammed Patel'
__contributor__ = 'Xinwwei chen, Fernando Pena Cantu,Javier Turnes, Eddie Park'
__copyright__ = ['university of waterloo']
__contact__ = ['m32patel@uwaterloo.ca', 'xinweic@uwaterloo.ca']
__version__ = '1.0.0'
__date__ = '2024-04-05'

import torch
from torch import nn
import torch.nn.functional as F


class OrderedCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super(OrderedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, output: torch.Tensor, target: torch.Tensor):

        criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=self.ignore_index)
        loss = criterion(output, target)
        # calculate the hard predictions by using softmax followed by an argmax
        softmax = torch.nn.functional.softmax(output, dim=1)
        hard_prediction = torch.argmax(softmax, dim=1)
        # set the mask according to ignore index
        mask = target == self.ignore_index
        hard_prediction = hard_prediction[~mask]
        target = target[~mask]
        # calculate the absolute difference between target and prediction
        weights = torch.abs(hard_prediction-target) + 1
        # remove ignored index losses
        loss = loss[~mask]
        # if done normalization with weights the loss becomes of the order 1e-5
        # loss = (loss * weights)/weights.sum()
        loss = (loss * weights)
        loss = loss.mean()

        return loss


class MSELossFromLogits(nn.Module):
    def __init__(self, chart, ignore_index=-100):
        super(MSELossFromLogits, self).__init__()
        self.ignore_index = ignore_index
        self.chart = chart
        if self.chart == 'SIC':
            self.replace_value = 11
            self.num_classes = 12
        elif self.chart == 'SOD':
            self.replace_value = 4
            self.num_classes = 5
        elif self.chart == 'FLOE':
            self.replace_value = 7
            self.num_classes = 8
        else:
            raise NameError(f'The chart {self.chart} is not recognized')
        
        # Create class weights for expectation calculation: [0, 1, 2, ..., N-1]
        self.register_buffer('class_weights', torch.arange(self.num_classes).float())

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        """
        Calculate MSE/Distance loss treating the classification problem as an ordinal regression.
        Instead of One-Hot MSE, we calculate the expected class value from the softmax distribution
        and compare it to the ground truth class index.
        """
        # Create a mask for valid pixels (not ignore_index)
        valid_mask = (target != self.ignore_index)  # (B, H, W)

        # If no valid pixels, return zero loss
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=output.device, requires_grad=True)

        # Select valid pixels
        # output shape: (B, C, H, W) -> permute to (B, H, W, C) -> mask to (N_valid, C)
        output_valid = output.permute(0, 2, 3, 1)[valid_mask] # (N_valid, num_classes)
        target_valid = target[valid_mask].float()             # (N_valid,)

        # Calculate Softmax probabilities
        probs = F.softmax(output_valid, dim=1)  # (N_valid, num_classes)

        # Calculate Expected Value (Soft Prediction)
        # E[x] = sum(p_i * i)
        # self.class_weights shape: (num_classes,)
        # Ensure class_weights is on the same device
        pred_expected = torch.sum(probs * self.class_weights.to(output.device), dim=1) # (N_valid,)

        # Calculate MSE between Expected Value and True Class Index
        loss = F.mse_loss(pred_expected, target_valid)

        return loss

class GCELoss(nn.Module):
    """Generalized Cross-Entropy Loss，对噪声标签具有鲁棒性。

    L = (1 - p_y^q) / q
    q=1 等价于 CE；q→0 等价于 MAE；推荐 q=0.7。

    注意：GCE 在训练初期梯度较弱（p_y 小时梯度趋近 0），从零训练时可能导致
    收敛缓慢或崩溃。建议配合 warmup_epochs 使用：前 N epoch 用纯 CE，
    之后切换到 GCE，或直接使用 label_smoothing 的 CrossEntropyLoss。

    Args:
        q:             噪声鲁棒参数，q ∈ (0, 1]，越小越鲁棒。
        weight:        各类别权重，list/tuple 或 1-D FloatTensor [C]。
        ignore_index:  忽略该标签值的像素。
        warmup_epochs: 前 N epoch 使用纯 CE loss（q=1），之后切换到 GCE。
                       调用方需在每 epoch 开始时调用 set_epoch(epoch)。
    """
    def __init__(self, q: float = 0.7, weight=None, ignore_index: int = 255,
                 warmup_epochs: int = 10):
        super().__init__()
        self.q = q
        self.ignore_index = ignore_index
        self.warmup_epochs = warmup_epochs
        self._current_epoch = 0
        if weight is not None:
            self.register_buffer('weight', torch.FloatTensor(weight))
        else:
            self.weight = None

    def set_epoch(self, epoch: int):
        """在每个 epoch 开始时调用，控制 warmup 阶段。"""
        self._current_epoch = epoch

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid_mask = targets != self.ignore_index
        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0

        probs = F.softmax(inputs, dim=1)
        p_y = probs.gather(1, targets_safe.unsqueeze(1)).squeeze(1)   # [B, H, W]

        # warmup 阶段用纯 CE（等价于 q=1）
        if self._current_epoch < self.warmup_epochs:
            loss = -torch.log(p_y.clamp(min=1e-7))
        else:
            loss = (1.0 - p_y ** self.q) / self.q

        if self.weight is not None:
            loss = loss * self.weight[targets_safe]

        loss = loss * valid_mask.float()
        return loss.sum() / valid_mask.float().sum().clamp(min=1.0)


class FocalLoss(nn.Module):
    """Alpha-weighted Focal Loss for multi-class segmentation.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: focusing parameter >= 0. gamma=0 degenerates to weighted CE.
        weight: per-class alpha weights, list/tuple or 1-D FloatTensor [C].
        ignore_index: pixels with this label are excluded from loss.
    """

    def __init__(self, gamma: float = 2.0, weight=None, ignore_index: int = 255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        if weight is not None:
            self.register_buffer('weight', torch.FloatTensor(weight))
        else:
            self.weight = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs : [B, C, H, W] logits
        # targets: [B, H, W]   class indices
        valid_mask = targets != self.ignore_index          # [B, H, W]
        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0                      # prevent gather OOB

        log_prob = F.log_softmax(inputs, dim=1)            # [B, C, H, W]
        log_pt = log_prob.gather(1, targets_safe.unsqueeze(1)).squeeze(1)  # [B, H, W]
        pt = log_pt.exp()

        if self.weight is not None:
            alpha_t = self.weight[targets_safe]            # [B, H, W]
        else:
            alpha_t = inputs.new_ones(targets.shape)

        loss = -alpha_t * (1.0 - pt) ** self.gamma * log_pt   # [B, H, W]
        loss = loss * valid_mask.float()
        return loss.sum() / valid_mask.float().sum().clamp(min=1.0)


class WaterConsistencyLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.keys = ['SIC', 'SOD', 'FLOE']
        self.activation = nn.Softmax(dim=1)

    def forward(self, output):
        # 需要至少3个任务才能计算跨任务水体一致性；单任务模式直接返回0
        available = [k for k in self.keys if k in output]
        if len(available) < 3:
            return torch.tensor(0.0, device=next(iter(output.values())).device)
        sic = self.activation(output[available[0]])[:, 0, :, :]
        sod = self.activation(output[available[1]])[:, 0, :, :]
        floe = self.activation(output[available[2]])[:, 0, :, :]
        return torch.mean((sic-sod)**2 + (sod-floe)**2 + (floe-sic)**2)

# only applicable to regression outputs
class MSELossWithIgnoreIndex(nn.MSELoss):
    def __init__(self, ignore_index=255, reduction='mean'):
        super(MSELossWithIgnoreIndex, self).__init__(reduction=reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        mask = (target != self.ignore_index).type_as(input)
        diff = input.squeeze(-1) - target
        diff = diff * mask
        loss = torch.sum(diff ** 2) / mask.sum()
        return loss


class AsymmetricSoftLabelLoss(nn.Module):
    """Cross-entropy with cost-matrix-derived soft labels.

    For true class c, the training target is:
        q_j = softmax(-temperature * C[c, :])_j

    High-cost predictions (large C[c][j]) receive near-zero target probability;
    the true class (C[c][c]=0) receives the highest. This changes *what the
    model is asked to output* (a cost-aware soft distribution), in contrast to
    MERL which changes *how the loss gradient is weighted*.

    temperature > 1 → sharper targets (closer to one-hot)
    temperature < 1 → softer targets (more mass on adjacent classes)

    Args:
        cost_matrix: [K, K] float tensor, C[true_class, pred_class]
        temperature: controls label softness (default 1.0)
        ignore_index: label value excluded from loss
    """

    def __init__(self, cost_matrix: torch.Tensor, temperature: float = 1.0,
                 ignore_index: int = 255):
        super().__init__()
        self.ignore_index = ignore_index
        # Soft label matrix: row c = target distribution when true class is c
        soft_targets = F.softmax(-temperature * cost_matrix.float(), dim=1)  # [K, K]
        self.register_buffer('soft_targets', soft_targets)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.float()
        targets = targets.long()

        valid_mask  = targets != self.ignore_index          # [B, H, W]
        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0

        log_probs = F.log_softmax(inputs, dim=1)            # [B, K, H, W]

        # q[b, h, w, :] = soft_targets[true_class, :]      [B, H, W, K]
        q = self.soft_targets[targets_safe]

        # Soft cross-entropy: -sum_j q_j * log p_j
        loss = -(q * log_probs.permute(0, 2, 3, 1)).sum(dim=-1)  # [B, H, W]

        loss = loss * valid_mask.float()
        return loss.sum() / valid_mask.float().sum().clamp(min=1.0)


class OrdinalBrierScoreLoss(nn.Module):
    """Weighted Ordinal Brier Score Loss (Cramér distance with cost-derived weights).

    L = (1/N) * sum_i sum_{k=0}^{K-2} w_k * (CDF_pred(k) - 1[y_i <= k])^2

    This is the L2 analogue of EMDLoss (which uses L1). The squared penalty
    creates stronger gradient signals for large CDF deviations.

    Boundary weights w_k are derived from the cost matrix: for each threshold k
    (between class k and k+1), w_k = average cost of all errors that cross it.
    This makes the dangerous water↔ice boundary the most penalised.

    This is a proper scoring rule for ordinal distributions (Epstein 1969),
    which EMDLoss is not — the unique minimiser is always the true distribution.

    Args:
        num_classes: number of ordinal classes K
        cost_matrix: [K, K] float tensor used to derive boundary weights.
                     If None, uses uniform weights (standard OBS).
        ignore_index: label value excluded from loss
    """

    def __init__(self, num_classes: int, cost_matrix: torch.Tensor = None,
                 ignore_index: int = 255):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        K = num_classes
        if cost_matrix is not None:
            C = cost_matrix.float()
            raw_w = []
            for k in range(K - 1):
                costs = []
                for i in range(k + 1, K):       # true > k (underestimation errors)
                    for j in range(0, k + 1):
                        costs.append(C[i, j].item())
                for i in range(0, k + 1):        # true <= k (overestimation errors)
                    for j in range(k + 1, K):
                        costs.append(C[i, j].item())
                raw_w.append(sum(costs) / len(costs))
            w = torch.tensor(raw_w, dtype=torch.float32)
            # Normalise: weights sum to K-1 so total loss scale matches unweighted OBS
            w = w / w.sum() * (K - 1)
        else:
            w = torch.ones(K - 1, dtype=torch.float32)

        self.register_buffer('threshold_weights', w)        # [K-1]

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.float()
        targets = targets.long()

        valid_mask  = targets != self.ignore_index          # [B, H, W]
        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0

        probs    = F.softmax(inputs, dim=1)                 # [B, K, H, W]
        cdf_pred = torch.cumsum(probs, dim=1)[:, :-1, :, :]  # [B, K-1, H, W]

        k_range  = torch.arange(self.num_classes - 1, device=inputs.device)  # [K-1]
        # CDF_true(k) = 1[y <= k]
        cdf_true = (targets_safe.unsqueeze(1) <= k_range.view(1, -1, 1, 1)).float()

        w    = self.threshold_weights.view(1, -1, 1, 1)    # [1, K-1, 1, 1]
        loss = (w * (cdf_pred - cdf_true) ** 2).sum(dim=1) # [B, H, W]

        loss = loss * valid_mask.float()
        return loss.sum() / valid_mask.float().sum().clamp(min=1.0)


class CORALLoss(nn.Module):
    """Loss for the CORAL ordinal output head.

    Expects inputs as P(y > k) sigmoid probabilities produced by CORALHead,
    NOT raw logits.  Applies cost-matrix-derived threshold weights (same scheme
    as OrdinalBrierScoreLoss) but uses BCE instead of MSE, matching the CORAL
    binary classification framing.

    inputs  : (B, K-1, H, W) — P(y > k) from CORALHead, values in (0, 1)
    targets : (B, H, W)      — integer class labels in [0, K-1]

    Asymmetric thresholds (risk-aware training): ``under_alpha`` (α ≥ 1) scales
    ONLY the positive-class BCE term t_k·(-log p_k), i.e. the cost of failing to
    cross a threshold the true label is above = underestimating ice severity.
    This shifts every ordinal decision boundary from p_k = 0.5 to p_k* =
    1/(1+α), training a risk-averse (prefer-higher-ice) classifier intrinsically.
    α = 1 recovers the standard symmetric CORAL loss.
    """

    def __init__(self, num_classes: int, cost_matrix: torch.Tensor = None,
                 ignore_index: int = 255, under_alpha: float = 1.0):
        super().__init__()
        self.K = num_classes
        self.ignore_index = ignore_index
        self.under_alpha = float(under_alpha)
        K = num_classes
        if cost_matrix is not None:
            C = cost_matrix.float()
            raw_w = []
            for k in range(K - 1):
                costs = []
                for i in range(k + 1, K):
                    for j in range(0, k + 1):
                        costs.append(C[i, j].item())
                for i in range(0, k + 1):
                    for j in range(k + 1, K):
                        costs.append(C[i, j].item())
                raw_w.append(sum(costs) / len(costs))
            w = torch.tensor(raw_w, dtype=torch.float32)
            w = w / w.sum() * (K - 1)
        else:
            w = torch.ones(K - 1, dtype=torch.float32)
        self.register_buffer('threshold_weights', w)  # (K-1,)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.float()  # prevent float16 rounding 1-ε → 1.0 → log(0) = -Inf → NaN
        targets = targets.long()
        valid_mask = targets != self.ignore_index
        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0

        # Binary target for each threshold k: 1[y > k]
        k_range = torch.arange(self.K - 1, device=inputs.device)
        target_k = (targets_safe.unsqueeze(1) > k_range.view(1, -1, 1, 1)).float()  # [B, K-1, H, W]

        p = inputs.clamp(1e-6, 1 - 1e-6)
        # Asymmetric: under_alpha weights the positive (underestimation) term only.
        bce = -(self.under_alpha * target_k * p.log()
                + (1 - target_k) * (1 - p).log())                      # [B, K-1, H, W]

        w = self.threshold_weights.view(1, -1, 1, 1)
        loss = (w * bce).sum(dim=1)   # [B, H, W]
        loss = loss * valid_mask.float()
        return loss.sum() / valid_mask.float().sum().clamp(min=1.0)


class MatrixExpectedRiskLoss(nn.Module):
    """Matrix Expected Risk Loss (MERL).

    Aligns training directly with an arbitrary cost matrix C[true, pred].
    L = E_{j ~ p}[C[true][j]] = sum_j p_j * C[true][j]

    Gradient: dL/d_logit_j = p_j * (C[true][j] - sum_k p_k * C[true][k])
    Models are guided to concentrate probability on low-cost classes.

    Args:
        cost_matrix: [K, K] float tensor, C[true_class, pred_class]
        ignore_index: label value excluded from loss computation
    """

    def __init__(self, cost_matrix: torch.Tensor, ignore_index: int = 255):
        super().__init__()
        self.ignore_index = ignore_index
        self.register_buffer('cost_matrix', cost_matrix.float())  # [K, K]

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Force fp32: prevents float16 gradient underflow under autocast.
        inputs = inputs.float()
        targets = targets.long()

        valid_mask  = targets != self.ignore_index         # [B, H, W]
        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0                      # prevent OOB index

        probs = torch.softmax(inputs, dim=1)               # [B, K, H, W]

        # cost_row[b, h, w] = C[true[b,h,w], :]  shape [B, H, W, K]
        cost_row = self.cost_matrix[targets_safe]          # [B, H, W, K]

        # expected risk per pixel: sum_j p_j * C[true][j]
        expected_risk = (probs.permute(0, 2, 3, 1) * cost_row).sum(dim=-1)  # [B, H, W]

        expected_risk = expected_risk * valid_mask.float()
        return expected_risk.sum() / valid_mask.float().sum().clamp(min=1.0)


class CostSensitiveCrossEntropyLoss(nn.Module):
    """Cost-Sensitive Cross Entropy Loss (CSCS).

    L = -log(p_true) + sum_j C[true][j] * (-log(1 - p_j))

    The first term is standard CE: pushes p_true UP.
    The second term is a cost-weighted binary CE penalty: for each class j,
    (-log(1-p_j)) is minimised when p_j→0, so high-cost wrong classes are
    pushed DOWN proportional to their cost. Because C[true][true]=0, the
    true class contributes nothing to the penalty — no masking needed.

    Gradient direction:
        ∂L/∂z_j  (j≠true) ≈ p_j·(1 - C[true][j]/(1-p_j))  — negative for high cost
        ∂L/∂z_true         — positive (CE + interaction from penalty)

    Difference from MERL:
        MERL : L = sum_j p_j * C[true][j]           — expected cost (linear in p)
        CSCS : CE + cost-weighted binary penalty      — logarithmic in (1-p_j)

    Note: for small p_j (early training), -log(1-p_j) ≈ p_j, so CSCS ≈ CE + MERL.

    Args:
        cost_matrix:  [K, K] float tensor, C[true_class, pred_class]
        ignore_index: label value excluded from loss computation
    """

    def __init__(self, cost_matrix: torch.Tensor, ignore_index: int = 255):
        super().__init__()
        self.ignore_index = ignore_index
        self.register_buffer('cost_matrix', cost_matrix.float())  # [K, K]

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.float()
        targets = targets.long()

        valid_mask   = targets != self.ignore_index          # [B, H, W]
        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0

        probs = torch.softmax(inputs, dim=1)                 # [B, K, H, W]

        # CE term: -log(p_true)
        true_log_p = torch.log(
            probs.gather(1, targets_safe.unsqueeze(1)).squeeze(1).clamp(min=1e-7)
        )                                                    # [B, H, W]
        ce_loss = -true_log_p

        # Cost-weighted binary CE penalty: sum_j C[true][j] * (-log(1 - p_j))
        # C[true][true] = 0, so true class contributes 0 automatically.
        log_1mp = torch.log((1.0 - probs).clamp(min=1e-7))  # [B, K, H, W]
        cost_row = self.cost_matrix[targets_safe]            # [B, H, W, K]
        penalty  = -(cost_row * log_1mp.permute(0, 2, 3, 1)).sum(dim=-1)  # [B, H, W]

        loss = (ce_loss + penalty) * valid_mask.float()
        return loss.sum() / valid_mask.float().sum().clamp(min=1.0)


class MixedLoss(nn.Module):
    """Weighted linear combination of multiple loss functions.

    L = sum_i weights[i] * losses[i](inputs, targets)

    Primary use case: CE (preserves F1) + ordinal risk loss (improves NRS).
        L = 1.0 * CrossEntropyLoss + λ * OrdinalBrierScoreLoss

    Args:
        losses:  list of nn.Module, the constituent losses (must be non-empty)
        weights: list of float, one per loss.  Defaults to uniform 1.0.
    """

    def __init__(self, losses: list, weights: list = None):
        super().__init__()
        if not losses:
            raise ValueError('MixedLoss requires at least one constituent loss.')
        self.loss_fns = nn.ModuleList(losses)
        if weights is None:
            weights = [1.0] * len(losses)
        if len(weights) != len(losses):
            raise ValueError(f'weights length {len(weights)} != losses length {len(losses)}')
        self.weights = list(float(w) for w in weights)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.float()
        total = torch.zeros((), device=inputs.device, dtype=inputs.dtype)
        for fn, w in zip(self.loss_fns, self.weights):
            total = total + w * fn(inputs, targets)
        return total


class EMDLoss(nn.Module):
    """Earth Mover's Distance (Wasserstein-1) loss for ordinal classification.

    EMD(P, Q) = Σ_{k=0}^{K-2} |CDF_P(k) - CDF_Q(k)|

    For one-hot true label at class c, CDF_Q is a step function:
        CDF_Q(k) = 0 if k < c, else 1.

    This naturally penalizes distant errors more than adjacent ones:
    adjacent error (|pred-true|=1) → cost=1; two-away error → cost=2; etc.
    Equivalent to the ordinal cost matrix with cost[i][j] = |i-j|.

    Args:
        num_classes: number of ordinal classes K
        ignore_index: label value to exclude from loss computation
    """

    def __init__(self, num_classes: int, ignore_index: int = 255):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs:  [B, C, H, W] logits
        # targets: [B, H, W]   class indices
        # Force fp32: F.softmax / cumsum are float16-eligible under autocast, causing
        # gradient underflow without GradScaler. CE avoids this (PyTorch promotes it to
        # fp32 automatically), so we must do the same explicitly here.
        inputs = inputs.float()
        targets = targets.long()

        valid_mask = targets != self.ignore_index           # [B, H, W]
        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0                       # prevent out-of-bounds gather

        probs = F.softmax(inputs, dim=1)                    # [B, C, H, W]

        # Predicted CDF: cumulative sum over class dim; drop the last entry (always 1)
        cdf_pred = torch.cumsum(probs, dim=1)[:, :-1, :, :]   # [B, K-1, H, W]

        # True CDF: step function — CDF_true(k) = 1 if k >= true_class, else 0
        # k ranges over 0..K-2
        class_range = torch.arange(self.num_classes - 1, device=inputs.device)  # [K-1]
        cdf_true = (class_range.view(1, -1, 1, 1) >= targets_safe.unsqueeze(1)).float()  # [B, K-1, H, W]

        # EMD per pixel = L1 distance between the two CDFs summed over class cuts
        emd = torch.abs(cdf_pred - cdf_true).sum(dim=1)    # [B, H, W]

        emd = emd * valid_mask.float()
        return emd.sum() / valid_mask.float().sum().clamp(min=1.0)
