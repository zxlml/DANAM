#!/bin/bash
# Train DANAM on the simulated data of the paper (Fig. 2 scenarios).

pip install -r requirements.txt

cd "$(dirname "$0")/model" || exit 1

# Four simulation scenarios (DANAM) + sparsity ablation (DANAM-, lam=0)
python DANAM_main.py --dataset Simulated --perturbation clean      --lam 1e-2 --n_repeats 10
python DANAM_main.py --dataset Simulated --perturbation outliers   --lam 1e-2 --n_repeats 10
python DANAM_main.py --dataset Simulated --perturbation mislabeled --lam 1e-2 --n_repeats 10
python DANAM_main.py --dataset Simulated --perturbation imbalanced --lam 1e-2 --n_repeats 10
python DANAM_main.py --dataset Simulated --perturbation clean      --lam 0    --n_repeats 10

# Real benchmark datasets
python DANAM_main.py --dataset Glioma --lam 1e-2 --n_repeats 10
python DANAM_main.py --dataset Telco  --lam 1e-2 --n_repeats 10
