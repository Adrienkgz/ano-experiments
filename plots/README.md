# Plot utilities

This directory provides lightweight utilities to **explore and visualise raw training logs** produced by the experiments described in our arXiv paper. The included scripts are designed to encourage reproducibility and facilitate further analysis beyond what is presented in the manuscript.

Our visualisations and tables represent only one perspective on the experimental data. These tools are intended to help readers examine results from multiple angles and adapt the analysis to their own research needs.

> **Note**: All scripts are self-contained and designed to be easily modifiable. Dependencies are minimal, and the logic is kept simple to encourage experimentation.

---

## Overview of scripts

| Script                  | Original name         | Description                                                                 | Example usage |
|-------------------------|------------------------|-----------------------------------------------------------------------------|----------------|
| `glue_table.py`         | `glue_tabs.py`         | Reconstructs the full GLUE benchmark table (per-task scores + macro average), as shown in the paper. | `python glue_table.py logs/NLP/GLUE --save results.md` |
| `training_curves.py`    | `plot_training_ev...`  | Plots the evolution of a chosen metric across optimisers and seeds. Supports EMA smoothing, min/max bands, and downsampling. | `python training_curves.py logs/CIFAR10 --metric test_acc` |
| `optimizer_summary.py`  | `show_perf_recap.py`   | Summarises each optimiser’s performance by computing mean and standard deviation of a selected metric. Optionally averages over the final 10 epochs. | `python optimizer_summary.py logs/Ant-v5 --metric mean_reward_last_n_eps` |

You may rename the scripts to these more explicit names for clarity. The original names are retained here for backward compatibility.

---

## Quick start

```bash
# Install required dependencies
pip install numpy pandas matplotlib tabulate

# Reproduce the GLUE benchmark table
python plots/glue_tabs.py

# Plot training curves
python plots/training_curves

# Summarise optimiser performance on HalfCheetah-v5
python plots/perf_summary.py
