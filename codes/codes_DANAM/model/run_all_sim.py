# -*- coding: utf-8 -*-
"""Sequential runner for all DANAM experiments (survives tool timeouts)."""
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))

RUNS = [
    # (perturbation, lam, repeats)
    ('clean', '1e-2', 10),
    ('outliers', '1e-2', 10),
    ('mislabeled', '1e-2', 10),
    ('imbalanced', '1e-2', 10),
    ('clean', '0', 10),  # DANAM- ablation (no group sparsity)
]

env = dict(os.environ, OMP_NUM_THREADS='2', PYTHONUNBUFFERED='1')

for pert, lam, reps in RUNS:
    cmd = [sys.executable, os.path.join(HERE, 'DANAM_main.py'),
           '--dataset', 'Simulated', '--perturbation', pert,
           '--lam', lam, '--n_repeats', str(reps),
           '--dropout', '0.1',
           '--logdir', os.path.join(HERE, '..', 'logs'),
           '--save_plot', 'true' if lam != '0' else 'false']
    print('RUNNING:', ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=HERE, env=env, check=True)
print('ALL DONE', flush=True)
