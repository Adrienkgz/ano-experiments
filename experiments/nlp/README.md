# 🧠 NLP Experiments — GLUE Benchmark

This folder contains all experiments related to the evaluation of **ANO** and baseline optimizers on NLP tasks using transformer models.

We follow the standard **GLUE benchmark** (excluding WNLI), covering 8 sentence- and sentence-pair classification tasks.  
The model architecture used is **BERT-base**, and all training runs adhere to common fine-tuning conventions.

---

## 📁 Structure

```
├── glue/
│   ├── logs/            # JSON logs in <Optimizer>-seed<Seed>.json format
```

---

## 📐 Metrics

We follow the official **GLUE evaluation metrics** for each task:

- Accuracy (`acc`) for SST-2, MNLI, etc.
- Matthews correlation (`matthews_corrcoef`) for CoLA
- F1 or average of metrics (e.g., `0.5*(Acc+F1)`) for QQP, MRPC, STS-B, etc.

To recreate the GLUE score table as shown in the paper, use:

```bash
python plots/glue_table.py
```

This script is located in the [`/plots`](../../plots) directory.

---

## 🚧 Note on availability

The results for **QQP** are currently **being re-run** to ensure alignment with official GLUE metrics.  
Other tasks are complete and can be fully reproduced from the logs provided. All the seeds aren't available.

---

## 🧪 Usage

All logs follow the format:

```python
<Optimizer>-seed<Seed>.json
```

You can use the plotting utilities in [`/plots`](../../plots) to:

- Compare training dynamics
- Summarise final metrics (mean ± std)
- Build custom tables

---

## 📎 Tips

- All scripts are designed for **reproducibility** and reuse
- To add a new optimizer, duplicate one of the existing configs and register it in the launcher
- Intermediate metrics are logged at each epoch for easy post-analysis

---
