# 🧪 Experiments

This folder contains all experiments accompanying the paper  
**“ANO: Faster is Better in Noisy Landscape”**.

Each subdirectory corresponds to a major domain evaluated in the study:

```
├── computer-vision/           # CIFAR-10 and ImageNet-100 classification benchmarks
├── nlp/                       # GLUE tasks (BERT), text classification
├── drl/                       # Deep Reinforcement Learning (MuJoCo + SAC)
├── hyperparameters_tuning/   # Scripts and results for hyperparameter tuning
```

Logs from our runs are included when available. All training scripts can be re-run to regenerate the results.  
Each folder is self-contained, with clear entrypoints for reproducibility.

---

## 📊 Data exploration and analysis

To visualise metrics, generate tables, or compare optimisers:

👉 Use the scripts in [`/plots`](../plots)  
They allow you to:

- Reconstruct benchmark tables (e.g. GLUE)
- Plot metric curves with smoothing and confidence intervals
- Summarise final performance (mean ± std) per optimiser

---

## 🔄 Ongoing experiments

Not all runs are complete — some experiments are still being processed and logs will be added progressively.

We aim for full reproducibility and transparency. If you wish to contribute results or run additional benchmarks, feel free to open a pull request or reach out.

---

## 💡 Domains at a glance

| Folder                  | Tasks & Datasets                            | Notes |
|-------------------------|---------------------------------------------|-------|
| `computer-vision/`      | CIFAR-10, ImageNet-100                      | ResNet based scripts |
| `nlp/`                  | GLUE benchmark, BERT fine-tuning           | GLUE |
| `drl/`                  | MuJoCo (Ant-v5, HalfCheetah-v5, etc.)      | SAC agent |
| `hyperparameters_tuning/` | Tuning logs + scripts for all domains     | Grid/random search with logging |

---

For more details on each setup, see the `README.md` inside each subfolder (when available).
