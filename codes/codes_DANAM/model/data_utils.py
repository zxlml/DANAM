# coding=utf-8
"""Data utilities for DANAM.

Includes:
* Simulated data generation following the paper (Fig. 2):
    y_i = sign( sum_{j=1}^{6} f*_j(x_j) ),   x_i ~ U([0, 1]^p)
  with three perturbation scenarios: 10% outliers; 10% mislabeled samples;
  and imbalanced class distributions at a 1:10 ratio (r1 / r2).
* Real benchmark dataset loaders (Telco, Glioma).
* 8:1:1 train/validation/test splitting (Section 4.1 of the paper).
"""

import os.path as osp
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DATA_PATH = osp.join(osp.dirname(osp.abspath(__file__)), '..', 'dataset')
DatasetType = Tuple[np.ndarray, np.ndarray]

# ---------------------------------------------------------------------------
# True component functions f*_j of the simulated data, read from Fig. 2 of
# the paper:
#   f1 = (1/3) log(40x + 1)          f4 = 6 (x - 1/4)^3
#   f2 = -(5/3) exp(-2|x|)           f5 = 3 exp(-(2x - 1)^2)
#   f3 = exp(x^2)                    f6 = log(1 + x^2)
#
# NOTE on label generation: the caption states y = sign(sum_j f*_j(x_j)),
# but the printed functions satisfy sum_j f*_j >= 0.34 everywhere on
# [0,1]^6 (f3 >= 1 and f5 >= 1.1 always dominate f2 >= -1.67), which would
# make y = +1 a.s. Since an additive model is only identifiable up to
# per-component constants, the standard (and only consistent) reading is
# that each component is centered before summation. We verified this yields
# ~52%/48% class balance, matching the paper's balanced clean setting.
# ---------------------------------------------------------------------------
TRUE_FUNCTIONS: List[Tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
    ('(1/3)log(40x+1)',        lambda x: np.log(40.0 * x + 1.0) / 3.0),
    ('-(5/3)exp(-2|x|)',       lambda x: -5.0 / 3.0 * np.exp(-2.0 * np.abs(x))),
    ('exp(x^2)',               lambda x: np.exp(x ** 2)),
    ('6(x-1/4)^3',             lambda x: 6.0 * (x - 0.25) ** 3),
    ('3exp(-(2x-1)^2)',        lambda x: 3.0 * np.exp(-(2.0 * x - 1.0) ** 2)),
    ('log(1+x^2)',             lambda x: np.log(1.0 + x ** 2)),
]

# Per-component means over [0, 1] (analytic grid means), used to center each
# f*_j before generating labels.
_X_GRID = np.linspace(0.0, 1.0, 20001)
TRUE_FUNCTION_MEANS: List[float] = [
    float(np.mean(f(_X_GRID))) for _, f in TRUE_FUNCTIONS
]


def latent_score(x: np.ndarray, centered: bool = True) -> np.ndarray:
    """Computes sum_{j=1}^{6} f*_j(x_j) for inputs in [0, 1]^p.

    With ``centered=True`` (default) each component is shifted by its mean
    over [0, 1] before summation, which is required for ``y = sign(score)``
    to produce two classes (see note above).
    """
    score = np.zeros(x.shape[0], dtype=np.float64)
    for j, (_, f) in enumerate(TRUE_FUNCTIONS):
        comp = f(x[:, j])
        if centered:
            comp = comp - TRUE_FUNCTION_MEANS[j]
        score += comp
    return score


def generate_simulated_data(n_samples: int = 2000,
                            n_informative: int = 6,
                            n_noise: int = 4,
                            perturbation: str = 'clean',
                            r1: float = 0.10,
                            r2: int = 10,
                            seed: int = 42) -> Dict[str, np.ndarray]:
    """Generates the simulated classification data of the paper (Fig. 2).

    Args:
      n_samples: total number of samples to draw.
      n_informative: number of informative features (paper: 6).
      n_noise: number of independent noise features (x ~ U([0,1]^p)).
      perturbation: one of {'clean', 'outliers', 'mislabeled', 'imbalanced'}.
      r1: fraction of training samples corrupted (outliers / mislabels).
      r2: majority:minority ratio for the 'imbalanced' scenario (paper: 10).
      seed: random seed.

    Returns:
      Dict with keys 'x', 'y' (in {-1,+1}), 'train_idx', 'val_idx', 'test_idx'.
    """
    rng = np.random.default_rng(seed)
    p = n_informative + n_noise
    x = rng.uniform(0.0, 1.0, size=(n_samples, p))
    y = np.sign(latent_score(x))
    # Avoid ties at zero.
    y[y == 0] = 1.0
    y = y.astype(np.float64)

    # 8:1:1 split (train/val/test), stratified-ish via shuffling.
    perm = rng.permutation(n_samples)
    n_train = int(0.8 * n_samples)
    n_val = int(0.1 * n_samples)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    # ----- Perturbations are applied to the training set only (Sec. 4.1) ----
    if perturbation == 'outliers':
        # 10% of the training inputs are replaced by gross outliers far
        # outside the support [0, 1]^p.
        n_out = int(np.ceil(r1 * n_train))
        out_rows = rng.choice(train_idx, size=n_out, replace=False)
        x[out_rows] = rng.uniform(-5.0, 5.0, size=(n_out, p))
    elif perturbation == 'mislabeled':
        # 10% of the training labels are flipped to the next category.
        n_flip = int(np.ceil(r1 * n_train))
        flip_rows = rng.choice(train_idx, size=n_flip, replace=False)
        y[flip_rows] = -y[flip_rows]
    elif perturbation == 'imbalanced':
        # Create a 1:10 majority:minority imbalance in the training set by
        # down-sampling the positive class (the clean data are roughly
        # balanced, so keeping all negatives and reducing positives to
        # 1/10th of their count yields the target ratio).
        tr_labels = y[train_idx]
        pos_idx = train_idx[tr_labels > 0]
        neg_idx = train_idx[tr_labels < 0]
        n_pos_keep = max(1, int(round(len(neg_idx) / float(r2))))
        keep_pos = rng.choice(pos_idx, size=min(n_pos_keep, len(pos_idx)),
                              replace=False)
        train_idx = np.concatenate([neg_idx, keep_pos])
        rng.shuffle(train_idx)
    elif perturbation != 'clean':
        raise ValueError('Unknown perturbation: {}'.format(perturbation))

    return {'x': x.astype(np.float32), 'y': y.astype(np.float32),
            'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx}


def min_max_scale(x_train: np.ndarray,
                  *others: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Scales features to [-1, 1] using statistics of the training set."""
    x_min = x_train.min(axis=0, keepdims=True)
    x_max = x_train.max(axis=0, keepdims=True)
    span = np.where(x_max - x_min < 1e-12, 1.0, x_max - x_min)

    def _scale(x):
        return (2.0 * (x - x_min) / span - 1.0).astype(np.float32)

    return tuple([_scale(x_train)] + [_scale(x) for x in others])


# ---------------------------------------------------------------------------
# Real benchmark datasets
# ---------------------------------------------------------------------------

def load_telco_churn_data() -> Dict[str, np.ndarray]:
    """Loads the Telco Customer Churn dataset (n=7043, p=21)."""
    df = pd.read_csv(osp.join(DATA_PATH, 'telcom_churn.csv'), sep=',', header=0)
    x_df = df[df.columns[:-1]].copy()
    if x_df['TotalCharges'].dtype == object:
        x_df['TotalCharges'] = x_df['TotalCharges'].replace(' ', 0).astype('float64')
    x_df = pd.get_dummies(x_df, dummy_na=True).astype('float32')
    y = df[df.columns[-1]].values.astype(np.float32)
    return {'x': x_df.values.astype(np.float32), 'y': y,
            'column_names': list(x_df.columns)}


def load_glioma_data() -> Dict[str, np.ndarray]:
    """Loads the Glioma Grading Clinical dataset (n=839, p=23)."""
    df = pd.read_csv(osp.join(DATA_PATH, 'TCGA_InfoWithGrade.csv'), header=0)
    x_df = df[df.columns[:-1]].copy()
    x_df = pd.get_dummies(x_df, dummy_na=True).astype('float32')
    y = df[df.columns[-1]].values.astype(np.float32)
    return {'x': x_df.values.astype(np.float32), 'y': y,
            'column_names': list(x_df.columns)}


def split_indices(y: np.ndarray, n_samples: int, seed: int,
                  train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Random 8:1:1 split of indices (paper Sec. 4.1)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    n_train = int(train_ratio * n_samples)
    n_val = int(val_ratio * n_samples)
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]


def load_dataset(dataset_name: str, seed: int = 42, **sim_kwargs) -> Dict:
    """Unified dataset loading interface.

    Returns dict with float32 'x' (features scaled to [-1,1] on train stats)
    and 'y' in {0,1}, plus index splits.
    """
    if dataset_name == 'Simulated':
        data = generate_simulated_data(seed=seed, **sim_kwargs)
        x, y, y_signed = data['x'], data['y'], data['y']
        tr, va, te = data['train_idx'], data['val_idx'], data['test_idx']
    elif dataset_name == 'Telco':
        d = load_telco_churn_data()
        x, y = d['x'], d['y']
        y_signed = np.where(y > 0, 1.0, -1.0)
        tr, va, te = split_indices(y, len(y), seed)
    elif dataset_name == 'Glioma':
        d = load_glioma_data()
        x, y = d['x'], d['y']
        y_signed = np.where(y > 0, 1.0, -1.0)
        tr, va, te = split_indices(y, len(y), seed)
    else:
        raise ValueError('{} not found!'.format(dataset_name))

    x_tr, x_va, x_te = min_max_scale(x[tr], x[va], x[te])
    return {
        'x_train': x_tr, 'y_train': np.where(y_signed[tr] > 0, 1.0, 0.0),
        'x_val': x_va, 'y_val': np.where(y_signed[va] > 0, 1.0, 0.0),
        'x_test': x_te, 'y_test': np.where(y_signed[te] > 0, 1.0, 0.0),
        'n_features': x.shape[1],
    }
