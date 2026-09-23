# coding=utf-8
"""Main entry for training and testing DANAM.

Examples
--------
# Simulated data (Fig. 2 scenarios), DANAM vs. DANAM- ablation:
python DANAM_main.py --dataset Simulated --perturbation clean      --lam 1e-2
python DANAM_main.py --dataset Simulated --perturbation outliers   --lam 1e-2
python DANAM_main.py --dataset Simulated --perturbation mislabeled --lam 1e-2
python DANAM_main.py --dataset Simulated --perturbation imbalanced --lam 1e-2
python DANAM_main.py --dataset Simulated --perturbation clean      --lam 0   # DANAM-

# Real datasets:
python DANAM_main.py --dataset Glioma --lam 1e-2
python DANAM_main.py --dataset Telco  --lam 1e-2
"""

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data_utils
import DANAM_train
import graph_builder
import models


def str2bool(v):
    return str(v).lower() in ('1', 'true', 'yes', 'y')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='DANAM')
    parser.add_argument('--dataset', default='Simulated',
                        choices=['Simulated', 'Telco', 'Glioma'])
    parser.add_argument('--perturbation', default='clean',
                        choices=['clean', 'outliers', 'mislabeled', 'imbalanced'],
                        help='Simulated-data scenario (paper Fig. 2).')
    parser.add_argument('--n_samples', type=int, default=2000)
    parser.add_argument('--n_informative', type=int, default=6,
                        help='Informative features of simulated data (paper: 6).')
    parser.add_argument('--n_noise', type=int, default=4,
                        help='Independent noise features of simulated data.')
    parser.add_argument('--r1', type=float, default=0.10,
                        help='Rate of corrupted training samples (outliers/mislabels).')
    parser.add_argument('--r2', type=int, default=10,
                        help='Majority:minority ratio for the imbalanced scenario.')
    # ----- model -----
    parser.add_argument('--num_basis_functions', type=int, default=64)
    parser.add_argument('--units_multiplier', type=int, default=2)
    parser.add_argument('--activation', default='exu', choices=['exu', 'relu'])
    parser.add_argument('--shallow', type=str2bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--feature_dropout', type=float, default=0.0)
    # ----- optimization (paper Sec. 4.1) -----
    parser.add_argument('--loss', default='eml',
                        choices=['eml', 'edf_kde', 'logistic'],
                        help='eml: EDF/EML loss of the paper (default); '
                             'edf_kde: literal Eq. 8 KDE estimator; '
                             'logistic: Eq. 2 baseline.')
    parser.add_argument('--warmup_iters', type=int, default=1000,
                        help='Logistic-to-EML loss annealing iterations.')
    parser.add_argument('--lam', type=float, default=1e-2,
                        help='Sparsity coefficient lambda in {1e-4,1e-3,1e-2,1e-1}.')
    parser.add_argument('--bandwidth', type=float, default=0.4,
                        help='Bandwidth percentage u in {0.4, 0.3, 0.2, 0.1} '
                             '(selected by 5-fold CV in the paper).')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size b in {32, 64, 128}.')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_iterations', type=int, default=10000,
                        help='Maximum iteration count T = 1e4.')
    parser.add_argument('--lr_decay_power', type=float, default=0.5)
    parser.add_argument('--eval_every', type=int, default=200)
    parser.add_argument('--patience', type=int, default=20,
                        help='Early-stopping patience (number of evaluations).')
    # ----- experiment protocol -----
    parser.add_argument('--n_repeats', type=int, default=10,
                        help='Number of experiment repetitions (paper: 20).')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--logdir', default='../logs')
    parser.add_argument('--save_plot', type=str2bool, default=True,
                        help='Save shape-function recovery plots (Fig. 2).')
    parser.add_argument('--verbose', type=str2bool, default=False)
    return parser.parse_args(argv)


def create_model(n_features: int, args) -> models.NAM:
    return models.NAM(
        num_inputs=n_features,
        num_units=args.num_basis_functions,
        shallow=args.shallow,
        dropout=args.dropout,
        feature_dropout=args.feature_dropout,
        activation=args.activation)


def run_single(args, seed: int, plot_info: Optional[dict] = None) -> dict:
    """Trains and evaluates DANAM on one data split. Returns metrics."""
    data = data_utils.load_dataset(
        args.dataset, seed=seed,
        n_samples=args.n_samples,
        n_informative=args.n_informative,
        n_noise=args.n_noise,
        perturbation=args.perturbation,
        r1=args.r1, r2=args.r2) if args.dataset == 'Simulated' \
        else data_utils.load_dataset(args.dataset, seed=seed)

    torch.manual_seed(seed)
    model = create_model(data['n_features'], args)
    loss_fn = {'eml': graph_builder.eml_loss,
               'edf_kde': graph_builder.edf_kde_loss,
               'logistic': graph_builder.logistic_loss}[args.loss]
    result = DANAM_train.train_model(
        model,
        data['x_train'], data['y_train'],
        data['x_val'], data['y_val'],
        loss_fn=loss_fn,
        lam=args.lam,
        bandwidth=args.bandwidth,
        warmup_iters=args.warmup_iters,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_iterations=args.max_iterations,
        lr_decay_power=args.lr_decay_power,
        eval_every=args.eval_every,
        patience=args.patience,
        seed=seed,
        verbose=args.verbose)

    # Final test evaluation with the best-validation model.
    test_metrics = DANAM_train.evaluate(model, data['x_test'], data['y_test'])
    train_metrics = DANAM_train.evaluate(model, data['x_train'], data['y_train'])

    if plot_info is not None and args.dataset == 'Simulated':
        plot_info['model'] = model
    return {'train': train_metrics, 'val': result['val'], 'test': test_metrics}


def plot_shape_functions(args, model, path: str) -> None:
    """Fig. 2 style plot: true vs. learned component functions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_informative = min(args.n_informative, len(data_utils.TRUE_FUNCTIONS))
    grid = np.linspace(0.0, 1.0, 200, dtype=np.float32)
    ncols = 3
    nrows = int(np.ceil(n_informative / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)
    model.eval()
    with torch.no_grad():
        for j in range(n_informative):
            xj = torch.as_tensor(grid.reshape(-1, 1), dtype=torch.float32)
            fx = model.feature_nns[j](xj, training=False).cpu().numpy()
            ax = axes[j]
            name, f_true = data_utils.TRUE_FUNCTIONS[j]
            # Center both curves on the grid for comparability (additive
            # components are only identifiable up to constants).
            fx = fx - fx.mean()
            true_y = f_true(grid)
            true_y = true_y - true_y.mean()
            corr = float(np.corrcoef(fx, true_y)[0, 1])
            ax.plot(grid, true_y, 'k--', lw=1.5, label='True $f^*_{}$'.format(j + 1))
            ax.plot(grid, fx, 'r-', lw=1.5, label='DANAM $f_{}$'.format(j + 1))
            ax.set_title('{}  (r={:.2f})'.format(name, corr), fontsize=9)
            ax.legend(fontsize=7)
            ax.set_xlabel('$x_{}$'.format(j + 1), fontsize=8)
    for j in range(n_informative, len(axes)):
        axes[j].axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('Shape-function plot saved to', path)


def aggregate_and_report(results, tag: str) -> dict:
    summary = {}
    print(f'\n===== Results: {tag} ({len(results)} repeats) =====')
    for split in ['train', 'test']:
        for metric in ['MacroF1', 'ErrorRate', 'Accuracy']:
            vals = [r[split][metric] for r in results]
            m, s = float(np.mean(vals)), float(np.std(vals))
            summary[f'{split}_{metric}'] = (m, s)
            print(f'  {split:5s} {metric:10s}: {m:.4f} +- {s:.4f}')
    return summary


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.logdir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Dataset={args.dataset} perturbation={args.perturbation} '
          f'lam={args.lam} bandwidth={args.bandwidth} device={device}')
    if args.dataset == 'Simulated':
        print('True component functions:',
              [name for name, _ in data_utils.TRUE_FUNCTIONS])

    results = []
    plot_info = {} if (args.save_plot and args.dataset == 'Simulated') else None
    t0 = time.time()
    for rep in range(args.n_repeats):
        seed = args.seed + rep
        res = run_single(args, seed, plot_info=plot_info)
        results.append(res)
        print(f'[rep {rep + 1}/{args.n_repeats}] '
              f'train F1={res["train"]["MacroF1"]:.4f} err={res["train"]["ErrorRate"]:.4f} | '
              f'test F1={res["test"]["MacroF1"]:.4f} err={res["test"]["ErrorRate"]:.4f}')
    print(f'Total time: {time.time() - t0:.1f}s')

    tag = f'{args.dataset}_{args.perturbation}_lam{args.lam}_u{args.bandwidth}'
    summary = aggregate_and_report(results, tag)

    if plot_info:
        plot_shape_functions(args, plot_info['model'],
                             os.path.join(args.logdir,
                                          f'shape_{args.perturbation}.png'))

    # Save summary to CSV.
    csv_path = os.path.join(args.logdir, 'results.csv')
    header = 'setting,' + ','.join(k for k in summary.keys())
    row = tag + ',' + ','.join(f'{m:.4f}+-{s:.4f}' for m, s in summary.values())
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', encoding='utf-8') as f:
        if write_header:
            f.write(header + '\n')
        f.write(row + '\n')
    print('Summary appended to', csv_path)


if __name__ == '__main__':
    main()
