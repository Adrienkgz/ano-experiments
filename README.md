# 🚀 ANO: Faster is Better in Noisy Landscape

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16422081.svg)](https://doi.org/10.5281/zenodo.16422081)
[![PyPI](https://img.shields.io/pypi/v/ano-optimizer.svg)](https://pypi.org/project/ano-optimizer/)
[![Stars](https://img.shields.io/github/stars/Adrienkgz/ano-experiments?style=social)](https://github.com/Adrienkgz/ano-experiments)

**Official repository** for the paper:  
👉 **[ANO: Faster is Better in Noisy Landscape](https://zenodo.org/records/16422081)**

---

## ✨ Abstract

Stochastic optimizers are central to deep learning, yet widely used methods like Adam and Adan exhibit performance degradation in non-stationary or noisy environments, partly due to their reliance on momentum-based magnitude estimates.

We introduce **Ano**, a novel optimizer that decouples the direction and magnitude of parameter updates: momentum is applied exclusively to directional smoothing, while the step size uses instantaneous gradient magnitudes.

This design improves robustness to gradient noise while retaining the simplicity and efficiency of first-order methods. We also propose **Anolog**, a variant of Ano that dynamically adjusts the momentum coefficient β₁, effectively expanding the momentum window over time.

📈 Empirically, Ano demonstrates competitive or superior performance across three major domains — computer vision, natural language processing, and deep reinforcement learning.

🧪 Theoretical convergence guarantees and full logs are provided to support **open, reproducible research**.

---

## 📦 Install the optimiser library

The **Ano optimiser is available as a standalone Python library** via PyPI:

```bash
pip install ano-optimizer
```

👉 GitHub repo of the library: [github.com/username/ano-lib](https://github.com/Adrienkgz/ano-optimizer)

It supports direct usage with **PyTorch** and **Tensorflow**.  
Documentation and examples are available in the [library repo](https://github.com/Adrienkgz/ano-optimizer).

---

## 🗂 Repository layout

```
├── optimizers/               # Implementation of Ano and baseline optimizers
├── experiments/              # Training scripts and logs for all benchmarks
│   ├── computer-vision/
│   ├── drl/
│   ├── nlp/
│   └── hyperparameters_tuning/
├── analysis/                 # Additional analyses and ablation studies
├── plots/                    # Tools for visualisation and summarisation
├── utils/                    # Shared helpers
├── data/                     # Datasets (filled with datas from some training code)
├── LICENSE
└── README.md
```

---

## 🚀 Getting started

```bash
# Clone the repository
git clone https://github.com/username/ano-optimizer.git
cd ano-optimizer

# Install dependencies
pip install requirements.txt
```

All experiments are fully reproducible and come with logs for direct analysis.  
See the `plots/` folder to visualise training metrics, summarise performance, or recreate benchmark tables.

---

## 📊 Reproducibility

We provide all experimental logs and training scripts in this repository:

- Logs follow the naming convention `<Optimizer>-seed<Seed>.json`
- Scripts can be re-run to regenerate results and plots
- Data is automatically downloaded when needed (e.g., CIFAR-10)

Please refer to each subfolder (e.g., `experiments/drl/`) for specific details.

If you explore additional metrics or configurations, feel free to share your results via issues or PRs 🙌

---

## 📚 Citation

If you use this work, please cite the paper:

```plaintext
@misc{kegreisz2025ano,
  author       = {Kegreisz, Adrien},
  title        = {Ano: Faster is Better in Noisy Landscapes},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.16422081},
  url          = {https://doi.org/10.5281/zenodo.16422081}
}
```

---

## 🧪 Status and ongoing work

This repository is actively maintained, and some experiments are still ongoing. As a result, **not all logs or results are currently available**. We are also working on improving the repository's structure and readability to ensure a better user experience. Thank you for your patience as we continue to refine and enhance this project.

The current version of the paper is a **preprint** and may evolve based on feedback and new experiments. We believe in continuous improvement and welcome any contribution that can help make the work stronger — whether it concerns:

- the theoretical formulation or assumptions,
- empirical results or missing benchmarks,
- clarity of writing and reproducibility,
- or implementation details in the code.

📬 If you notice something unclear, missing, or improvable, please don’t hesitate to open an issue or start a discussion — we would love your feedback.

---

## 📬 Contact

For questions or collaborations, feel free to open an issue or reach out via email (see paper).

---


## 🤝 arXiv endorsement

We are in the process of submitting this work to arXiv.  
If you are an endorsed author and appreciate this work, **endorsements are warmly welcome** to help disseminate the research. Thank you 🙏

---

> ⭐ If you find this project useful, consider giving it a star to support the research!
