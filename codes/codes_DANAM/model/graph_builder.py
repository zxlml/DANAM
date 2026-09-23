# coding=utf-8
"""DANAM loss functions, optimization step and evaluation metrics.

Implements the algorithm of the paper:

* EDF (Estimated Density Function) loss for classification (Eq. 5 & 8):
    e_i   = 1 - sigmoid(y_i * f(x_i)),  y_i in {-1, +1}
    h_hat(e) = (1 / (n * u)) * sum_i K((e - e_i) / u)      (Eq. 4, KDE)
    L     = mean_i( -log h_hat(e_i) )
  where K is the Gaussian kernel and u is the bandwidth.

* Group-wise proximal (soft-thresholding) operator (Eq. 11):
    theta_j <- (1 - eta_t * lambda * sqrt(k_j) / ||s_j||_2)_+ * s_j
  applied to every subnetwork after each gradient step (Steps 1-3 of the
  optimization algorithm). This realizes the group-sparsity regularization
  alpha(Theta) in Eq. 7 and performs feature selection.
"""

import math
from typing import List

import numpy as np
import torch
from sklearn import metrics as sk_metrics

import models

KERNEL_CONST = 1.0 / math.sqrt(2.0 * math.pi)

# Lower clip of the variable bandwidth (same as the reference implementation).
H_MIN = 0.01


def edf_kde_loss(logits: torch.Tensor,
                 y_signed: torch.Tensor,
                 bandwidth: float = 0.1) -> torch.Tensor:
    """Literal EDF loss of Eq. 8 with batch KDE (Eq. 4).

    NOTE: for classification this estimator has a degenerate optimum (all
    errors collapse to a constant, e.g. f == 0 -> e == 0.5 maximizes the
    batch KDE density). It is kept for completeness/ablation; use `eml_loss`
    (the estimator of [13, 16] used by the paper's implementation) instead.
    """
    logits = logits.reshape(-1)
    y_signed = y_signed.reshape(-1).to(logits.dtype)
    errors = 1.0 - torch.sigmoid(y_signed * logits)  # in [0, 1]
    diff = (errors.unsqueeze(1) - errors.unsqueeze(0)) / bandwidth  # (m, m)
    kernel = KERNEL_CONST * torch.exp(-0.5 * diff * diff)
    h_hat = kernel.mean(dim=1) / bandwidth
    return -torch.log(h_hat + 1e-10).mean()


def variable_bandwidth(logits: torch.Tensor,
                       bandwidth_percentage: float) -> torch.Tensor:
    """Per-sample variable bandwidth from logit-space kNN distances [16].

    h_i is the distance to the k-th nearest neighbor among the mini-batch
    logits (k = bandwidth_percentage * m), clipped to [0.01, 1]. Samples in
    dense logit regions get a small bandwidth, isolated samples (potential
    outliers) get a large bandwidth and are automatically down-weighted.
    """
    logits = logits.reshape(-1, 1)
    m = logits.shape[0]
    k = max(1, int(float(m) * bandwidth_percentage))
    dist = torch.cdist(logits, logits)  # (m, m)
    knn = dist.topk(k, dim=1, largest=False).values[:, -1]
    return knn.clamp(H_MIN, 1.0)


def eml_loss(logits: torch.Tensor,
             y_signed: torch.Tensor,
             bandwidth: float = 0.1) -> torch.Tensor:
    """Expected Maximum-likelihood (EML) loss on the prediction errors.

    Implements the data-adaptive MLE objective of the paper (Eq. 3/5/8) as
    realized in the reference implementation [13, 16]: with the confidence
    error e_i = 1 - sigmoid(y_i * f(x_i)) and the variable bandwidth h_i
    estimated from the logit-space error density, the loss is the negative
    log expected likelihood over the mini-batch:

        L = -log( (1/m) * sum_i exp(-e_i / h_i) ).

    Misclassified / outlier samples receive a vanishing gradient, which
    yields the robustness studied in the paper. `bandwidth` is the
    bandwidth percentage u in {0.4, 0.3, 0.2, 0.1} (Sec. 4.1).
    """
    logits = logits.reshape(-1)
    y_signed = y_signed.reshape(-1).to(logits.dtype)
    errors = 1.0 - torch.sigmoid(y_signed * logits)  # in [0, 1]
    h = variable_bandwidth(logits.detach(), bandwidth)
    mean_likelihood = torch.mean(torch.exp(-errors / h))
    return -torch.log(mean_likelihood + 1e-8)


def edf_regression_loss(logits: torch.Tensor,
                        y_signed: torch.Tensor,
                        bandwidth: float = 0.1) -> torch.Tensor:
    """Literal EDF loss of Eq. 5 for regression-style targets.

    The density estimate is applied to the residuals e_i = y_i - f(x_i).
    Unlike the classification-confidence variant, this estimator is NOT
    degenerate: concentrating the residuals requires fitting the conditional
    mean, so it naturally bounds the model output and preserves the additive
    shape functions (used for the Fig. 2 shape-recovery study).
    """
    logits = logits.reshape(-1)
    y_signed = y_signed.reshape(-1).to(logits.dtype)
    residuals = y_signed - logits
    diff = (residuals.unsqueeze(1) - residuals.unsqueeze(0)) / bandwidth
    kernel = KERNEL_CONST * torch.exp(-0.5 * diff * diff)
    h_hat = kernel.mean(dim=1) / bandwidth
    return -torch.log(h_hat + 1e-10).mean()


def logistic_loss(logits: torch.Tensor, y_signed: torch.Tensor,
                  bandwidth: float = 0.1) -> torch.Tensor:
    """Classical logistic loss (Eq. 2), used as a baseline/ablation option."""
    return torch.nn.functional.softplus(-(y_signed.to(logits.dtype) * logits)).mean()


def group_proximal_step(nam_model: models.NAM, lr: float, lam: float) -> None:
    """In-place group soft-thresholding of every subnetwork (Eq. 11).

    Args:
      nam_model: trained NAM/DANAM model.
      lr: current step size eta_t.
      lam: sparsity regularization coefficient lambda.
    """
    if lam <= 0.0:
        return
    with torch.no_grad():
        for params in nam_model.feature_param_groups():
            theta = torch.cat([p.data.reshape(-1) for p in params])
            k_j = theta.numel()
            norm = torch.norm(theta, p=2)
            if norm.item() <= 1e-12:
                continue
            coef = 1.0 - lr * lam * math.sqrt(k_j) / norm.item()
            coef = max(0.0, float(coef))
            if coef != 1.0:
                for p in params:
                    p.data.mul_(coef)


def penalized_loss(loss_value: torch.Tensor,
                   nam_model: models.NAM,
                   l2_regularization: float = 0.0) -> torch.Tensor:
    """Optional auxiliary weight-decay penalty (kept from the NAM baseline)."""
    if l2_regularization > 0:
        num_networks = len(nam_model.feature_nns)
        l2 = sum(torch.sum(p ** 2) for p in nam_model.parameters())
        return loss_value + l2_regularization * l2 / num_networks
    return loss_value


def to_labels(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Converts sigmoid probabilities to binary labels in {0, 1}."""
    return (probs >= threshold).astype(np.int64)


def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """Computes Macro-F1, accuracy and error rate for binary labels {0, 1}."""
    y_pred = to_labels(probs)
    return {
        'MacroF1': float(sk_metrics.f1_score(y_true, y_pred, average='macro')),
        'Accuracy': float(sk_metrics.accuracy_score(y_true, y_pred)),
        'ErrorRate': float(1.0 - sk_metrics.accuracy_score(y_true, y_pred)),
    }
