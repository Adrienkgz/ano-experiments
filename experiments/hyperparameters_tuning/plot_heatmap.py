import os
import json
import math
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, LogNorm, LinearSegmentedColormap

FNAME_RE = re.compile(
    r"(?P<opt>[^_/]+)_lr(?P<lr>[0-9eE.+-]+)_b1(?P<b1>[0-9eE.+-]+)_b2(?P<b2>[0-9eE.+-]+)-seed(?P<seed>\d+)\.json"
)

def parse_filename(fname):
    m = FNAME_RE.match(os.path.basename(fname))
    if not m:
        raise ValueError(f"Nom de fichier non conforme : {fname}")
    return (
        m.group("opt"),
        float(m.group("lr")),
        float(m.group("b1")),
        float(m.group("b2")),
        int(m.group("seed")),
    )

def pick_metric(values, *, maximize: bool, use_last: bool):
    return values[-1] if use_last else (max(values) if maximize else min(values))

def extract_metric(path, metric, *, maximize: bool, use_last: bool):
    with open(path) as f:
        data = json.load(f)
    vals = [float(ep[metric]) for ep in data if metric in ep]
    if not vals:
        raise KeyError(f"{metric} absent de {path}")
    return pick_metric(vals, maximize=maximize, use_last=use_last)

def aggregate_by_opt_lr(logdir, metric, optimizers, maximize=True, use_last=False):
    all_buckets = defaultdict(list)
    for root, _, files in os.walk(logdir):
        for fn in files:
            if not fn.endswith(".json"): continue
            try:
                opt, lr, b1, b2, _ = parse_filename(fn)
            except ValueError:
                continue
            if opt not in optimizers: continue
            val = extract_metric(os.path.join(root, fn), metric, maximize=maximize, use_last=use_last)
            all_buckets[(opt, lr, b1, b2)].append(val)

    all_results = {}
    lrs_all = set()
    for opt in optimizers:
        lrs_opt = sorted({lr for (o, lr, _, _) in all_buckets if o == opt})
        lrs_all.update(lrs_opt)
        for lr in lrs_opt:
            b1s = sorted({b1 for (o, l, b1, _) in all_buckets if o == opt and l == lr})
            b2s = sorted({b2 for (o, l, _, b2) in all_buckets if o == opt and l == lr})
            mean = np.full((len(b2s), len(b1s)), np.nan)
            std = np.full_like(mean, np.nan)
            for (o, l, b1, b2), vs in all_buckets.items():
                if o != opt or l != lr: continue
                i, j = b2s.index(b2), b1s.index(b1)
                mean[i, j] = np.mean(vs)
                std[i, j] = np.std(vs, ddof=0)
            all_results[(opt, lr)] = (mean, std, b1s, b2s)
    lrs = sorted(lrs_all)
    return all_results, lrs

def soft_rdygn():
    # Palette RdYlGn avec un rouge adouci (mélange avec du gris clair)
    base_cmap = plt.get_cmap("RdYlGn", 256)
    colors = base_cmap(np.linspace(0, 1, 256))
    soft_red = np.array([0.85, 0.3, 0.3])   # Rouge adouci, tirant vers le brique
    # On remplace les 45 premiers (zones faibles) par un dégradé soft_red -> palette originale
    for i in range(45):
        ratio = i / 45
        colors[i, :3] = colors[i, :3] * ratio + soft_red * (1 - ratio)
    # Tu peux aussi adoucir le vert foncé si besoin, mais généralement il passe bien
    return LinearSegmentedColormap.from_list("SoftRdYlGn", colors, N=256)


def plot_custom_grid(
    all_results, optimizers, lrs, metric, maximize, use_last, outdir,
    main_title="Hyperparameter Grid Search — Optimizer × LR",
    global_scale=True, use_log=False, cmap=None
):
    if cmap is None:
        cmap = soft_rdygn()

    os.makedirs(outdir, exist_ok=True)
    n_opt = len(optimizers)
    n_lr = len(lrs)
    fig, axes = plt.subplots(
        n_opt, n_lr, figsize=(3.7 * n_lr, 3.5 * n_opt), sharey=False, sharex=False, layout="constrained"
    )
    if n_opt == 1: axes = axes[None, :]
    if n_lr == 1: axes = axes[:, None]

    # Colorbar
    if global_scale:
        vmin = min(np.nanmin(mean) for (mean, _, _, _) in all_results.values())
        vmax = max(np.nanmax(mean) for (mean, _, _, _) in all_results.values())
        norm = LogNorm(vmin=vmin, vmax=vmax) if use_log else Normalize(vmin, vmax)
    else:
        norm = None

        im_ref = None
    for i_opt, opt in enumerate(optimizers):
        for j_lr, lr in enumerate(lrs):
            ax = axes[i_opt, j_lr]
            key = (opt, lr)
            if key not in all_results:
                ax.axis('off')
                continue
            mean, std, b1s, b2s = all_results[key]
            _norm = norm or (LogNorm(*np.nanpercentile(mean, [1, 99])) if use_log else Normalize(*np.nanpercentile(mean, [1, 99])))
            im_ref = ax.imshow(mean, origin="lower", cmap=cmap, norm=_norm, aspect="auto")

            # Titre des colonnes (LR)
            if i_opt == 0:
                ax.set_title(f"lr={lr:.0e}", fontsize=14, fontweight="bold", pad=12)

            # Yticks et labels beta2 uniquement sur la colonne la plus à gauche
            if j_lr == 0:
                ax.set_yticks(range(len(b2s)))
                ax.set_yticklabels([f"{b2:g}" for b2 in b2s], fontsize=11)
                ax.set_ylabel(f"{opt}\nβ₂", fontsize=12, fontweight="bold")
            else:
                ax.set_yticks([])
                ax.set_yticklabels([])
                ax.set_ylabel("")

            # Xticks et labels beta1 sur toutes les colonnes
            if i_opt == n_opt - 1:
                ax.set_xlabel("β₁", fontsize=12, fontweight="bold")
            ax.set_xticks(range(len(b1s)))
            ax.set_xticklabels([f"{b1:g}" for b1 in b1s], rotation=45, fontsize=11)

            # Grille fine + annotations
            ax.set_xticks(np.arange(-0.5, len(b1s), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(b2s), 1), minor=True)
            ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.6, alpha=0.5)

            for i in range(len(b2s)):
                for j in range(len(b1s)):
                    val = mean[i, j]
                    sd = std[i, j]
                    txt = "–" if math.isnan(val) else f"{val:.2f}\n±{sd:.2f}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="black", fontweight="normal" if not math.isnan(val) else "normal")

            for spine in ax.spines.values():
                spine.set_linewidth(2)
                spine.set_color("#333333")

    if global_scale and im_ref is not None:
        cbar = fig.colorbar(im_ref, ax=axes, orientation="vertical", fraction=0.03, pad=0.04)
        cbar.set_label(metric, fontsize=13, fontweight="bold")
        cbar.ax.tick_params(labelsize=12)

    # Titre global bien positionné
    mode_txt = "last" if use_last else ("max" if maximize else "min")
    fig.suptitle(
        f"{main_title}", fontsize=17, fontweight="bold", y=1.06, x=0.47
    )
    plt.subplots_adjust(top=0.86)
    fname = os.path.join(outdir, f"ALL_{metric}_{mode_txt}_customgrid_ICLR.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK]", fname)

def main():
    LOGDIR = "experiments/hyperparameters_tuning/logs/halfcheetah"
    OUTDIR = "experiments/hyperparameters_tuning/heatmaps"
    OPTIMIZERS = ["AdamW", "Ano", "Adan", 'Lion', 'Grams', 'Anolog']

    METRIC = "mean_reward_last_n_eps" #mean_reward_last_n_eps
    MAXIMIZE = True
    USE_LAST = False

    GLOBAL_SCALE = True
    USE_LOG = False

    all_results, lrs = aggregate_by_opt_lr(
        LOGDIR, METRIC, OPTIMIZERS, maximize=MAXIMIZE, use_last=USE_LAST
    )
    plot_custom_grid(
        all_results, OPTIMIZERS, lrs, METRIC, MAXIMIZE, USE_LAST, OUTDIR,
        main_title="GridSearch",   # Mets ce que tu veux ici
        global_scale=GLOBAL_SCALE, use_log=USE_LOG,
    )

if __name__ == "__main__":
    main()
