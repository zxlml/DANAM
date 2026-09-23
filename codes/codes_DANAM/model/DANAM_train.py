# coding=utf-8
"""Training loop for DANAM (Section 3 of the paper).

Optimization algorithm (forward-backward splitting):
  Step 1. Estimate the gradient of the average task loss on a mini-batch b
          of size m:  g_t = (1/m) * sum_{i in b} grad_Theta L(x_i, y_i).
  Step 2. Gradient descent step:  S_t = Theta_t - eta_t * g_t.
  Step 3. Group-wise proximal operator of the sparsity regularizer (Eq. 11).

The learning rate follows a polynomial decay with power = 0.5 and a maximum
iteration count of T = 10^4 (Section 4.1, Parameter Selection).
"""

import copy
from typing import Callable, Dict, Optional

import numpy as np
import torch

import graph_builder


def evaluate(model, x: np.ndarray, y01: np.ndarray, batch_size: int = 4096) -> Dict[str, float]:
    """Evaluates Macro-F1 / accuracy / error rate on a dataset."""
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32)
            logits = model(xb, training=False)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    probs = np.concatenate(probs)
    model.train()
    return graph_builder.compute_metrics(y01, probs)


def train_model(model: torch.nn.Module,
                x_train: np.ndarray,
                y_train01: np.ndarray,
                x_val: np.ndarray,
                y_val01: np.ndarray,
                loss_fn: Callable = graph_builder.eml_loss,
                lam: float = 1e-2,
                bandwidth: float = 0.1,
                warmup_iters: int = 1000,
                learning_rate: float = 1e-3,
                batch_size: int = 64,
                max_iterations: int = 10000,
                lr_decay_power: float = 0.5,
                eval_every: int = 200,
                patience: int = 20,
                seed: int = 42,
                device: Optional[str] = None,
                verbose: bool = False) -> Dict:
    """Trains a DANAM model following the paper's optimization algorithm.

    Returns a dict with the best (by validation Macro-F1) metrics and state.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.train()

    x_all = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    # Labels in {-1, +1} for the EDF loss (Eq. 8).
    y_signed_all = torch.as_tensor(2.0 * y_train01 - 1.0, dtype=torch.float32, device=device)
    n_train = x_all.shape[0]
    steps_per_epoch = max(1, n_train // batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_f1 = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = {}
    best_train_metrics = {}
    bad_evals = 0
    rng = np.random.default_rng(seed)

    # Class-balanced mini-batch sampling (upsampling the rare class), which
    # realizes the balanced training stream of the reference implementation
    # and handles the imbalanced scenario (r2 = 1:10).
    pos_idx = np.where(y_train01 > 0.5)[0]
    neg_idx = np.where(y_train01 <= 0.5)[0]
    half = max(1, batch_size // 2)

    for t in range(1, max_iterations + 1):
        # Polynomial learning-rate decay (power = 0.5), Sec. 4.1.
        eta_t = learning_rate * max(0.0, 1.0 - (t - 1) / max_iterations) ** lr_decay_power
        for group in optimizer.param_groups:
            group['lr'] = eta_t

        # Step 1: mini-batch gradient of the average task loss (Eq. 9).
        b_pos = rng.choice(pos_idx, size=half, replace=len(pos_idx) < half)
        b_neg = rng.choice(neg_idx, size=batch_size - half, replace=len(neg_idx) < batch_size - half)
        batch_idx = np.concatenate([b_pos, b_neg])
        rng.shuffle(batch_idx)
        xb, yb = x_all[batch_idx], y_signed_all[batch_idx]
        logits = model(xb, training=True)
        # Loss annealing: the distribution-aware (EML/EDF) loss is a
        # bounded-influence loss whose confidence weighting is only meaningful
        # once the logits carry signal. We therefore anneal from the logistic
        # loss (Eq. 2) to the task loss over `warmup_iters` iterations.
        if warmup_iters > 0:
            alpha = min(1.0, (t - 1) / warmup_iters)
            loss = (1.0 - alpha) * graph_builder.logistic_loss(logits, yb, bandwidth) \
                + alpha * loss_fn(logits, yb, bandwidth)
        else:
            loss = loss_fn(logits, yb, bandwidth)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Step 2: gradient descent update  S_t = Theta_t - eta_t * g_t.
        optimizer.step()

        # Step 3: group soft-thresholding proximal operator (Eq. 11).
        if lam > 0 and hasattr(model, 'feature_param_groups'):
            graph_builder.group_proximal_step(model, eta_t, lam)

        if t % eval_every == 0 or t == max_iterations:
            val_metrics = evaluate(model, x_val, y_val01)
            if verbose:
                print(f'  iter {t:6d} | loss {loss.item():.4f} | '
                      f'val MacroF1 {val_metrics["MacroF1"]:.4f}')
            if val_metrics['MacroF1'] > best_val_f1:
                best_val_f1 = val_metrics['MacroF1']
                best_state = copy.deepcopy(model.state_dict())
                best_metrics = val_metrics
                best_train_metrics = evaluate(model, x_train, y_train01)
                bad_evals = 0
            else:
                bad_evals += 1
                if bad_evals >= patience:
                    if verbose:
                        print(f'  early stopping at iter {t}')
                    break

    model.load_state_dict(best_state)
    if not best_metrics:  # no validation improvement recorded yet
        best_metrics = evaluate(model, x_val, y_val01)
        best_train_metrics = evaluate(model, x_train, y_train01)
    return {
        'train': best_train_metrics,
        'val': best_metrics,
        'state_dict': best_state,
        'iterations': t,
    }
