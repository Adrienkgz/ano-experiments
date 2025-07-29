"""
Utility to summarise a chosen training metric across multiple optimisers.

This helper script accompanies the experiments reported in our arXiv
paper.  It consumes the raw JSON logs *as produced by the training
scripts* so that interested readers can revisit the unprocessed data
rather than only the visualisations included in the article.

Expected file-name convention
----------------------------
<Optimizer>-seed<Seed>.json      (e.g. ``AdamBaseline-seed1.json``)

Each file must contain a JSON list with one dictionary per epoch/step
and at least the following key:

* ``metric_key`` – name of the metric you want to aggregate
  (``train_acc``, ``mean_reward_last_n_eps``, …).

The main function :pyfunc:`analyze_optimizer_performance` aggregates
several seeds per optimiser and returns the mean and (population)
standard deviation of the selected metric.  An option allows you to
average over the *last 10 epochs* instead of taking the global best or
worst value.

The code is deliberately **simple and hackable** rather than a full-blown
library – feel free to tweak anything to fit your workflow.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Dict, List

import numpy as np
from tabulate import tabulate

__all__ = [
    "analyze_optimizer_performance",
    "show_table",
]


def analyze_optimizer_performance(
    folder_path: str,
    *,
    maximize: bool = True,
    remove_outliers: bool = False,
    make_a_mean_for_10_last_epoch: bool = False,
    metric_key: str = "train_acc",
) -> Dict[str, Dict[str, float]]:
    """Aggregate *metric_key* across all JSON logs found in *folder_path*.

    Parameters
    ----------
    folder_path : str
        Directory containing the ``*.json`` log files.
    maximize : bool, default ``True``
        If ``True`` keep the *maximum* value of *metric_key*; otherwise keep
        the minimum.  Ignored when *make_a_mean_for_10_last_epoch* is
        ``True``.
    remove_outliers : bool, default ``False``
        Discard the lowest and highest value **per optimiser** when at least
        three seeds are available.
    make_a_mean_for_10_last_epoch : bool, default ``False``
        Instead of the global best/worst value, compute the mean of the
        last 10 epochs.  If fewer than 10 points are present, the mean is
        taken over the available ones and a warning is emitted.
    metric_key : str, default ``'train_acc'``
        Key to look up inside each JSON entry.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping ``optimizer -> {'mean_<metric_key>': …, 'std_<metric_key>': …}``.
        When no data survive the filtering steps the corresponding values
        are ``np.nan``.

    Notes
    -----
    Only filenames following the pattern ``<Optimizer>-seed<Seed>.json`` are
    considered.
    """
    optimizer_values: Dict[str, List[float]] = defaultdict(list)

    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return {}

    filename_re = re.compile(r"(?P<opt>.+?)-seed\d+\.json$")
    for filename in os.listdir(folder_path):
        if not filename.endswith(".json"):
            continue

        m = filename_re.match(filename)
        if not m:
            print(f"Warning: Unrecognised filename '{filename}' (skipped).")
            continue
        optimizer_name = m["opt"]

        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Could not read '{filename}': {e} (skipped).")
            continue

        if not (isinstance(data, list) and data):
            print(f"Warning: '{filename}' is empty or malformed (skipped).")
            continue

        metric_values = [
            item[metric_key]
            for item in data
            if isinstance(item, dict) and metric_key in item
        ]

        if not metric_values:
            print(f"Warning: Key '{metric_key}' not found in '{filename}' (skipped).")
            continue

        # Select one value per seed
        if make_a_mean_for_10_last_epoch:
            last_values = metric_values[-10:]
            if len(last_values) < 10:
                print(
                    f"Warning for '{filename}': found only {len(last_values)} "
                    f"epoch(s). Averaging over the available points."
                )
            selected_value = float(np.mean(last_values))
        else:
            selected_value = float(max(metric_values) if maximize else min(metric_values))

        optimizer_values[optimizer_name].append(selected_value)

    # Aggregate per optimiser
    results: Dict[str, Dict[str, float]] = {}
    for optim, values in optimizer_values.items():
        if remove_outliers and len(values) > 2:
            ordered = sorted(values)
            values = ordered[1:-1]  # drop min and max

        if values:
            results[optim] = {
                f"mean_{metric_key}": float(np.mean(values)),
                f"std_{metric_key}": float(np.std(values)),
            }
        else:
            results[optim] = {
                f"mean_{metric_key}": float("nan"),
                f"std_{metric_key}": float("nan"),
            }
            print(f"Warning: No data retained for '{optim}'.")

    return results


def show_table(data: Dict[str, Dict[str, float]], title: str, metric_key: str) -> None:
    """Pretty-print *data* in a grid using **tabulate**."""
    print(f"\n{title}")
    if not data:
        print("No data to display.")
        return

    mean_key = f"mean_{metric_key}"
    std_key = f"std_{metric_key}"

    table_rows = []
    for opt, vals in data.items():
        mean_val = vals.get(mean_key)
        std_val = vals.get(std_key)
        table_rows.append(
            (
                opt,
                f"{mean_val:.3f}" if isinstance(mean_val, float) else "N/A",
                f"{std_val:.2f}" if isinstance(std_val, float) else "N/A",
            )
        )

    print(tabulate(table_rows, headers=["Optimiser", "Mean", "Std"], tablefmt="grid"))


# --------------------------------------------------------------------------- #
# Example usage
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    FOLDER = "experiments/drl/logs/HalfCheetah-v5"
    METRIC = "mean_reward_last_n_eps"

    perf_last10 = analyze_optimizer_performance(
        FOLDER,
        maximize=True,
        remove_outliers=False,
        make_a_mean_for_10_last_epoch=True,
        metric_key=METRIC,
    )

    perf_best = analyze_optimizer_performance(
        FOLDER,
        maximize=True,
        metric_key=METRIC,
    )

    show_table(perf_last10, f"==  Mean over last 10 epochs ({METRIC})  ==", METRIC)
    show_table(perf_best,   f"==  Best performance ({METRIC})          ==", METRIC)
