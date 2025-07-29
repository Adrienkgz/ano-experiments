import os
import re
import json
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from tabulate import tabulate

def collect_metrics(
    root_dir: str = "GLUE",
    optimizer_filter: Optional[List[str]] = None,
    remove_outliers: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics: Dict[str, Dict[str, List[float]]] = {}

    for dataset in sorted(os.listdir(root_dir)):
        dataset_path = os.path.join(root_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue

        for fname in os.listdir(dataset_path):
            m = re.match(rf"(.*?)-seed(\d+)\.json", fname)
            if not m:
                continue
            optimizer, seed_str = m.groups()

            if optimizer_filter and optimizer not in optimizer_filter:
                continue

            fpath = os.path.join(dataset_path, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                run_log: List[Dict[str, Any]] = json.load(f)

            if not run_log:
                continue
            
            scores_in_run = []
            for step in run_log:
                agg_metric_dict = step.get("aggregated_metric")
                if agg_metric_dict and isinstance(agg_metric_dict, dict) and agg_metric_dict.values():
                    score = list(agg_metric_dict.values())[0]
                    scores_in_run.append(score)
            
            if not scores_in_run:
                continue 
            
            val = max(scores_in_run)

            metrics.setdefault(optimizer, {}).setdefault(dataset, []).append(val)

    mean_df = {}
    std_df = {}
    for optimizer, ds_dict in metrics.items():
        mean_df[optimizer] = {}
        std_df[optimizer] = {}
        for dataset, values in ds_dict.items():
            clean_vals = sorted(values)
            if remove_outliers and len(clean_vals) > 2:
                clean_vals = clean_vals[1:-1]
            mean_df[optimizer][dataset] = np.mean(clean_vals) if clean_vals else np.nan
            std_df[optimizer][dataset] = np.std(clean_vals, ddof=1) if len(clean_vals) > 1 else np.nan

    mean_df = pd.DataFrame(mean_df).T
    std_df = pd.DataFrame(std_df).T

    mean_df["Average"] = mean_df.mean(axis=1, skipna=True)
    std_df["Average"] = std_df.mean(axis=1, skipna=True)

    mean_df.sort_index(inplace=True)
    std_df = std_df.reindex(mean_df.index)

    return mean_df, std_df

def format_table(mean_df: pd.DataFrame, std_df: pd.DataFrame, decimals: int = 2) -> str:
    headers = ["Optimizer"] + list(mean_df.columns)
    table = []

    for opt in mean_df.index:
        row = [opt]
        for col in mean_df.columns:
            mean_val = mean_df.loc[opt, col]
            std_val = std_df.loc[opt, col]
            if col == "Average":
                cell = f"{100*mean_val:.{decimals}f}" if pd.notna(mean_val) else "-"
            else:
                if pd.notna(mean_val):
                    if pd.notna(std_val):
                        cell = f"{100*mean_val:.{decimals}f}±{100*std_val:.{decimals}f}"
                    else:
                        cell = f"{100*mean_val:.{decimals}f}"
                else:
                    cell = "-"
            row.append(cell)
        table.append(row)

    return tabulate(table, headers=headers, tablefmt="github")


if __name__ == "__main__":
    mean_df, std_df = collect_metrics('experiments/nlp/logs')
    print(format_table(mean_df, std_df))
