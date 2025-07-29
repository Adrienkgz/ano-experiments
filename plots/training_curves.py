"""Utility to visualise training metrics across multiple optimisers.

This helper script accompanies the experiments of the corresponding
scientific paper. The JSON files produced by the training scripts are **not**
post-processed in any way; they are consumed as-is so that readers can
re-explore the raw data rather than only the views shown in the article.

Each run is expected to live in its own JSON file named
``<Optimizer>_seed<Seed>.json`` (e.g. ``AdamBaseline_seed1.json``).
Every file contains a list of dictionaries, one per epoch/step, with at
least these two keys:

* ``epoch`` (or ``total_steps`` for DRL) — x-axis value.
* The metric you are interested in (``test_acc``,
  ``aggregated_metric.Acc+F1``, ``mean_reward_last_n_eps``, …).

Functions below let you aggregate different seeds, smooth the curves with
an EMA, and compare several optimisers on the same plot.

The code is deliberately **simple and hackable** rather than being a fully
fledged library – feel free to tweak anything you need.
"""

# Imports
import os
import json
import re
from typing import Any, List, Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt

# Utils

def apply_ema(series: pd.Series, beta: float):
    """Exponential Moving Average (EMA) smoothing.

    Parameters
    ----------
    series : pd.Series
        The raw sequence.
    beta : float
        Smoothing coefficient in the open interval *(0, 1)*. The closer to 1,
        the smoother (but also more delayed) the curve.

    Returns
    -------
    pd.Series
        Smoothed sequence with the same index as *series*.
    """
    avg, smoothed = series.iloc[0], []
    for val in series:
        avg = beta * avg + (1 - beta) * val
        smoothed.append(avg)
    return pd.Series(smoothed, index=series.index)


def get_nested_value(d: dict, path: str):
    """Retrieve a nested value from *d* using dot notation.

    Example
    -------
    >>> d = {"aggregated_metric": {"Acc+F1": 0.93}}
    >>> get_nested_value(d, "aggregated_metric.Acc+F1")
    0.93

    If the full path cannot be resolved the function quietly returns
    ``None`` to make downstream filtering easier.

    Parameters
    ----------
    d : dict
        The dictionary to traverse.
    path : str
        Dot-separated path such as ``"aggregated_metric.Acc+F1"``.

    Returns
    -------
    Any or None
        The value found, or ``None`` if the path is missing.
    """
    cur: Any = d
    for key in path.split('.'):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return None
    return cur

# Plot helper

def plot(
    directory_path: str,
    metric_key: str = 'test_acc',
    epoch_key: str = 'epoch',
    font_size: int = 22,
    have_to_save: bool = False,
    output_path: str = 'experiments/',
    optimizer_filter: Optional[List[Tuple[str, str]]] = None,
    ema_beta: float = 0.0,
    show_min_max: bool = True,
    sample_every: Optional[int] = None,
):
    """Aggregate and plot *metric_key* for every optimiser found.

    Parameters
    ----------
    directory_path : str
        Directory containing the ``*.json`` logs.
    metric_key : str, default ``'test_acc'``
        Metric to plot. Supports dot notation for nested JSON keys.
    epoch_key : str, default ``'epoch'``
        Field to use on the x-axis (e.g. ``'total_steps'`` for DRL).
    font_size : int, default 22
        Base font size.
    have_to_save : bool, default ``False``
        Save the figure to *output_path* if ``True``.
    output_path : str, default ``'experiments/'``
        Folder where the PNG will be written when *have_to_save* is ``True``.
    optimizer_filter : list[tuple[str, str]] | None
        Optional mapping ``[(raw_prefix, label), ...]``. Only files whose
        *raw_prefix* is present will be kept and displayed as *label*.
    ema_beta : float, default 0.0
        EMA smoothing coefficient. Set to 0 to disable.
    show_min_max : bool, default ``True``
        Whether to draw a translucent min/max band across seeds.
    sample_every : int | None
        Down-sample the curves by keeping one point out of *sample_every*.

    Returns
    -------
    None
        Displays the Matplotlib figure and optionally saves it.
    """

    # Build a mapping <file_prefix -> label> if user supplied one
    if optimizer_filter:
        opt_map = {file: disp for file, disp in optimizer_filter}
        keep_only = set(opt_map)
    else:
        opt_map, keep_only = {}, None

    all_rows = []
    filename_re = re.compile(r'^(?P<opt>.+?)[-_]seed(?P<seed>\d+)', flags=re.I)

    # ——————————————— Read files
    try:
        filenames = [f for f in os.listdir(directory_path) if f.endswith('.json')]
    except FileNotFoundError:
        print(f"Erreur : dossier introuvable : «{directory_path}».")
        return
    if not filenames:
        print("Aucun fichier .json trouvé.")
        return

    for fname in filenames:
        m = filename_re.match(fname[:-5])
        if not m:
            print(f"Format inattendu: «{fname}» – ignoré.")
            continue

        optimizer, seed = m['opt'], int(m['seed'])
        if keep_only and optimizer not in keep_only:
            continue

        try:
            with open(os.path.join(directory_path, fname), encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Décodage JSON impossible : {fname} – ignoré.")
            continue
        if not isinstance(data, list):
            print(f"« {fname} » ne contient pas une liste JSON – ignoré.")
            continue

        for entry in data:
            if not isinstance(entry, dict) or epoch_key not in entry:
                continue
            metric_val = get_nested_value(entry, metric_key)
            if metric_val is None:
                continue

            all_rows.append({
                'optimizer': opt_map.get(optimizer, optimizer),
                'epoch': entry[epoch_key],
                'metric': metric_val,
                'seed': seed,
            })

    if not all_rows:
        print("Aucune donnée exploitable.")
        return

    # Build aggregated DataFrame
    df = pd.DataFrame(all_rows)
    agg = (
        df.groupby(['optimizer', 'epoch'])['metric']
          .agg(['mean', 'min', 'max'])
          .reset_index()
    )

    # Plot
    plt.figure(figsize=(12, 8))
    colors = [
        '#1E88E5', '#E53935', '#4CAF50', '#FFD600', '#7B1FA2',
        '#00ACC1', '#FF8F00', '#D81B60', '#795548', '#8BC34A'
    ]
    for i, opt in enumerate(agg['optimizer'].unique()):
        sub = agg[agg['optimizer'] == opt].sort_values('epoch')
        c = colors[i % len(colors)]

        if ema_beta > 0:
            sub['mean'] = apply_ema(sub['mean'], ema_beta)
            if show_min_max:
                sub['min'] = apply_ema(sub['min'], ema_beta)
                sub['max'] = apply_ema(sub['max'], ema_beta)

        if sample_every and sample_every > 1:
            sub = sub.iloc[::sample_every].reset_index(drop=True)

        plt.plot(sub['epoch'], sub['mean'], label=opt, linewidth=2, color=c)
        if show_min_max:
            plt.fill_between(sub['epoch'], sub['min'], sub['max'], alpha=0.2, color=c)

    # Aesthetics
    plt.xlabel(epoch_key.replace('_', ' ').title(), fontsize=font_size)
    y_label = metric_key.split('.')[-1].replace('_', ' ').title()
    plt.ylabel(y_label, fontsize=font_size)
    plt.legend(title="Optimiseur", fontsize=font_size, title_fontsize=1.1 * font_size)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if have_to_save:
        plt.savefig(os.path.join(output_path, 'optimizer_comparison.png'))
    plt.show()

# Examples
if __name__ == "__main__":
    """Minimal examples for the three families of experiments shipped with the paper.

    Adjust paths, metric names and optimiser lists to match your local
    directory layout.
    
    The tuple `(raw_prefix, label)` in `optimizer_filter` allows you to
    filter the optimisers you want to compare and rename them for display.
    The `ema_beta` parameter controls the smoothing of the curves.
    The `sample_every` parameter allows you to down-sample the curves
    by keeping one point out of every `sample_every` points.
    The `show_min_max` parameter controls whether to display a translucent
    band between the minimum and maximum values across seeds.
    The `have_to_save` parameter controls whether to save the plot as a PNG
    file in the specified `output_path`.
    The `output_path` parameter specifies the directory where the plot will be saved.
    If you want to save the plot, set `have_to_save` to `True` and specify
    a valid `output_path`. If you only want to display the plot without saving,
    set `have_to_save` to `False`.
    """

    # ====================================================================
    # Deep Reinforcement Learning (HalfCheetah‑v5)
    # ====================================================================
    plot(
        directory_path='experiments/drl/logs/HalfCheetah-v5',
        metric_key='mean_reward_last_n_eps',
        epoch_key='total_steps',
        optimizer_filter=[
            ('AdamBaseline', 'Adam'),
            ('AnoBaseline',  'Ano'),
            ('GramsBaseline', 'Grams'),
        ],
        show_min_max=True,
        have_to_save=False,
        output_path='plots/created_plots',
        ema_beta=0.0,
        sample_every=1,
    )

    # ====================================================================
    # Computer Vision (CIFAR‑10 classification)
    # ====================================================================
    plot(
        directory_path='experiments/computervision/logs/CIFAR10',
        metric_key='test_acc',
        epoch_key='epoch',
        optimizer_filter=[
            ('AdamBaseline', 'Adam'),
            ('RangerBaseline', 'Ranger'),
            ('SGD', 'SGD'),
        ],
        show_min_max=True,
        have_to_save=False,
        output_path='plots/created_plots',
        ema_beta=0.1,
        sample_every=3,
    )

    # ====================================================================
    # Natural Language Processing (IMDB sentiment analysis)
    # ====================================================================
    plot(
        directory_path='experiments/nlp/logs/IMDB',
        metric_key='aggregated_metric.Acc+F1',
        epoch_key='epoch',
        optimizer_filter=[
            ('AdamBaseline', 'Adam'),
            ('LionBaseline', 'Lion'),
            ('AdafactorBaseline', 'Adafactor'),
        ],
        show_min_max=True,
        have_to_save=False,
        output_path='plots/created_plots',
        ema_beta=0.05,
        sample_every=2,
    )
