<div align="center">

# 🧠 Distribution-Aware Neural Additive Models: Robust Interpretable Deep Learning with Feature Selection

[![Paper](https://img.shields.io/badge/Paper-ICASSP%202026-4C1FBD?logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/abstract/document/11463944)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-1F6FEB)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)

[English](README.md) | **简体中文**

</div>

> 🌟 如果你觉得 DANAM 对你有帮助，欢迎点亮一颗 **Star**！

## 📰 新闻

- **[2025-12]** 🔥 DANAM 被 **ICASSP 2026** 接收。
- **[2025-12]** 🚀 发布 PyTorch 实现，包含完整训练流程、仿真实验套件与形状函数恢复工具。

## 🧠 项目简介

神经可加模型（NAM）在可解释性与精度之间取得了良好平衡，但其依赖的经典损失函数对复杂噪声结构——离群点、标签噪声、类别不平衡——非常敏感。

**DANAM** 将学习过程建立在**自适应误差建模的最大似然估计**之上：

- 📈 **分布感知损失。** 通过**核密度估计（KDE）**在线估计经验误差分布，得到数据自适应的目标函数（EDF/EML 损失），无需施加苛刻的分布假设。
- ✂️ **自动特征选择。** 引入**组稀疏**正则与组软阈值**近端算子**，将无用特征的子网络整体置零，提升模型可解释性。
- 🛡️ **内在鲁棒性。** 在 10% 离群点、10% 标签翻转、1:10 类别不平衡等条件下依然保持高精度，而传统 logistic 损失的 NAM 会明显退化。
- 🔍 **可解释性。** 每个特征由一个专用子网络建模，其学习到的形状函数可直接可视化，并与真实函数对比。

## ✨ 核心亮点

- **EDF 损失**（论文式 8）：对概率预测误差进行密度估计 $\hat{h}\big(1-\sigma(y_i \sum_j f_j(x_i^j))\big)$，并采用变带宽的 EML 估计器实现。
- **三步优化**（论文第 3 节）：mini-batch 梯度 → 梯度下降步 → 组近端算子 $\theta_{j,t+1} = \big(1 - \eta_t\lambda\sqrt{k_j}/\|s_j\|_2\big)_+ s_j$。
- **忠实还原的仿真协议**：$y_i = \mathrm{sign}\big(\sum_{j=1}^{6} f^*_j(x_{ij})\big)$，采用图 2 的六个真实形状函数，并支持三种扰动场景（离群点 / 标签翻转 / 类别不平衡）。
- **形状函数恢复**：逐特征绘制学习曲线与真实曲线的对比，并给出 Pearson 相关诊断（图 2 风格）。

## 🚀 快速开始

### 1️⃣ 环境配置

```bash
conda create -n danam python=3.10 -y
conda activate danam
cd codes/codes_DANAM
pip install -r requirements.txt
```

<details>
<summary><b>依赖版本</b></summary>

| 包 | 版本 |
|---------|---------|
| torch | ≥ 2.0 |
| numpy | ≥ 1.24 |
| pandas | ≥ 2.0 |
| scikit-learn | ≥ 1.3 |
| matplotlib | ≥ 3.7 |

</details>

### 2️⃣ 运行完整仿真实验

```bash
sh run.sh
```

或单独运行各场景：

```bash
cd model

# 四种仿真场景（DANAM）+ 稀疏消融（DANAM-, lam = 0）
python DANAM_main.py --dataset Simulated --perturbation clean      --lam 1e-2 --n_repeats 10
python DANAM_main.py --dataset Simulated --perturbation outliers   --lam 1e-2 --n_repeats 10
python DANAM_main.py --dataset Simulated --perturbation mislabeled --lam 1e-2 --n_repeats 10
python DANAM_main.py --dataset Simulated --perturbation imbalanced --lam 1e-2 --n_repeats 10
python DANAM_main.py --dataset Simulated --perturbation clean      --lam 0    --n_repeats 10   # DANAM−

# 真实基准数据集（Glioma / Telco）
python DANAM_main.py --dataset Glioma --lam 1e-2 --n_repeats 10
python DANAM_main.py --dataset Telco  --lam 1e-2 --n_repeats 10
```

<details>
<summary><b>⚙️ 关键超参数（论文 4.1 节）</b></summary>

| 参数 | 默认值 | 说明 |
|----------|---------|-------------|
| `--lam` | `1e-2` | 稀疏系数 λ ∈ {1e-4, 1e-3, 1e-2, 1e-1} |
| `--bandwidth` | `0.4` | KDE 带宽比例 u ∈ {0.4, 0.3, 0.2, 0.1} |
| `--batch_size` | `64` | 批大小 b ∈ {32, 64, 128} |
| `--learning_rate` | `1e-3` | 初始学习率（多项式衰减，power 0.5） |
| `--max_iterations` | `10000` | 最大迭代次数 T = 10⁴ |
| `--warmup_iters` | `1000` | logistic→EML 损失退火迭代数 |
| `--loss` | `eml` | `eml`（默认）/ `edf_kde`（式 8 字面实现）/ `logistic`（式 2 基线） |
| `--n_repeats` | `10` | 实验重复次数（论文为 20） |

</details>

## 📁 项目结构

```
DANAM
├── codes/codes_DANAM
│   ├── model
│   │   ├── models.py          # NAM 架构（ExU 单元，逐特征子网络）
│   │   ├── graph_builder.py   # EDF / EML / logistic 损失 + 组近端算子
│   │   ├── DANAM_train.py     # 三步优化循环、类平衡采样
│   │   ├── data_utils.py      # 仿真数据生成（图 2）+ 真实数据集加载
│   │   ├── DANAM_main.py      # 命令行入口、指标汇总、图 2 风格绘图
│   │   └── run_all_sim.py     # 完整仿真实验顺序执行器
│   ├── dataset                # Glioma 与 Telco 基准数据集 CSV 文件
│   ├── logs                   # 实验汇总（results.csv）+ 形状函数图
│   ├── requirements.txt
│   └── run.sh
```

## 🙏 致谢

本项目基于 [NAM](https://arxiv.org/abs/2204.01513)（Neural Additive Models）以及 Chen 等人的 EDF 经验风险最小化框架（*Expert Systems with Applications*, 2024）构建，并感谢 [Sparse NAM](https://link.springer.com/chapter/10.1007/978-3-031-26409-2_32) 作者提供的组稀疏建模思路。

## 许可证

本项目采用 [Apache License 2.0](./LICENSE) 发布。

## 📖 引用

如果本仓库对你的研究有帮助，请引用（BibTeX 可从 [Google Scholar](https://scholar.google.com) 获取）：

```bibtex
@inproceedings{chen2026distributionaware,
  title        = {Distribution-Aware Neural Additive Models: Robust Interpretable Deep Learning with Feature Selection},
  author       = {Chen, Jingyi and Zhang, Xuelin and Yuan, Peipei and Liu, Liyuan and Chen, Hong},
  booktitle    = {ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages        = {1--5},
  year         = {2026},
  organization = {IEEE}
}
```
