import os, json, torch
from pathlib import Path

def save_logs(model_name, seed, history, folder=''):
    out_dir = Path(folder) / f"{model_name}-seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(f'{folder}/{model_name}-seed{seed}.json', 'w') as f:
        json.dump(history, f, indent=4)
    print("Logs saved.")