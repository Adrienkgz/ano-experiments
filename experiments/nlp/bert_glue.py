# Permit to run this script from any directory
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Imports
import torch, random, numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets, DatasetDict, interleave_datasets, get_dataset_split_names
from transformers import (
    BertConfig,
    AutoTokenizer,
    BertForMaskedLM,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_scheduler
)
from scipy.stats import pearsonr
from tqdm.auto import tqdm, trange
from utils import save_logs
import evaluate

# Reproducibility setup
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PJRT_DEVICE"] = "CUDA"
os.environ["XLA_USE_BF16"] = "0"
os.environ["XLA_DEVICE"] = "CUDA"
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# GLUE task information
GLUE_TASKS_INFO = {
    "sst2": {"num_labels": 2, "is_regression": False, "eval_split": "validation"},
    "cola": {"num_labels": 2, "is_regression": False, "eval_split": "validation"},
    "mrpc": {"num_labels": 2, "is_regression": False, "eval_split": "validation"},
    "stsb": {"num_labels": 1, "is_regression": True,  "eval_split": "validation"},
    "qqp":  {"num_labels": 2, "is_regression": False, "eval_split": "validation"},
    "mnli": {"num_labels": 3, "is_regression": False, "eval_split": "validation_matched"},
    "qnli": {"num_labels": 2, "is_regression": False, "eval_split": "validation"},
    "rte":  {"num_labels": 2, "is_regression": False, "eval_split": "validation"},
    "wnli": {"num_labels": 2, "is_regression": False, "eval_split": "validation"},
}
GLUE_EVAL_SPLITS = {
    "sst2": "validation", "cola": "validation", "mrpc": "validation",
    "stsb": "validation", "qqp": "validation", "qnli": "validation",
    "rte": "validation", "wnli": "validation",
    "mnli": ["validation_matched", "validation_mismatched"]
}

INPUT_COLUMNS = {
    "sst2": ("sentence", None),
    "cola": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
    "qqp":  ("question1", "question2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte":  ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

# Preprocessing
def preprocess_glue(tokenizer, raw_ds, task, max_length=128):
    col1, col2 = INPUT_COLUMNS[task]

    def tok_fn(batch):
        if col2 is None:
            enc = tokenizer(batch[col1], truncation=True, max_length=max_length)
        else:
            enc = tokenizer(batch[col1], batch[col2], truncation=True, max_length=max_length)
        enc["labels"] = batch["label"]
        return enc

    keep_cols = [c for c in raw_ds["train"].column_names if c == "label"]
    rm_cols = [c for c in raw_ds["train"].column_names if c not in keep_cols]
    return raw_ds.map(tok_fn, batched=True, remove_columns=rm_cols)

# Evaluation function
@torch.no_grad()
def evaluate_model(model, dataloader, device, task_name, is_reg):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, total_samples = 0, 0
    loss_fn = torch.nn.MSELoss() if is_reg else torch.nn.CrossEntropyLoss()

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"]
        outputs = model(**batch)
        logits = outputs.logits
        loss = loss_fn(logits.squeeze(-1) if is_reg else logits, labels.float() if is_reg else labels)
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
        if is_reg:
            all_preds.append(logits.squeeze(-1).cpu())
        else:
            all_preds.append(logits.argmax(dim=-1).cpu())
        all_labels.append(labels.cpu())

    avg_loss = total_loss / total_samples
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    metric_loader = evaluate.load("glue", "mnli" if task_name == "mnli" else task_name)
    raw_results_dict = metric_loader.compute(predictions=all_preds, references=all_labels)

    # Scoring
    if task_name == "cola":
        agg_metric, name = raw_results_dict["matthews_correlation"], "MCC"
    elif task_name == "stsb":
        agg_metric, name = (raw_results_dict["pearson"] + raw_results_dict["spearmanr"]) / 2.0, "Pearson+Spearman"
    elif task_name in ["mrpc", "qqp"]:
        agg_metric, name = (raw_results_dict["accuracy"] + raw_results_dict["f1"]) / 2.0, "Acc+F1"
    else: 
        agg_metric, name = raw_results_dict["accuracy"], "Accuracy"
    
    return avg_loss, agg_metric, name, raw_results_dict

# Training
def finetune_glue_task(
    task_name: str,
    seed_val: int,
    optimizer_cls: torch.optim.Optimizer,
    optimizer_name: str,
    *,
    num_epochs: int = 3,
    per_device_batch_size: int = 32,
    grad_accum_steps: int = 1,
    lr_scaling_rule: str = "linear",  # "linear" ou "none"
    warmup_ratio: float = 0.1,
    use_small_subset: bool = False,
    optimizer_params: dict[str, any] | None = None,
):
    base_learning_rate = optimizer_params['lr']
    optim_params = optimizer_params.copy()
    del optim_params['lr']
    set_seed(seed_val)
    device = torch.device('cuda')

    info = GLUE_TASKS_INFO[task_name]
    print(f"\n### FINETUNE {task_name} – seed={seed_val} – optimiser={optimizer_name}")
    raw = load_dataset("glue", task_name)
    if use_small_subset:
        raw["train"] = raw["train"].shuffle(seed=seed_val).select(range(5_000))
        if task_name == "mnli":
            raw["validation_matched"] = raw["validation_matched"].shuffle(seed=seed_val).select(range(200))
            raw["validation_mismatched"] = raw["validation_mismatched"].shuffle(seed=seed_val).select(range(200))
        else:
             raw[GLUE_EVAL_SPLITS[task_name]] = raw[GLUE_EVAL_SPLITS[task_name]].shuffle(seed=seed_val).select(range(200))

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds = preprocess_glue(tokenizer, raw, task_name)
    collator = DataCollatorWithPadding(tokenizer)

    train_dl = DataLoader(
        ds["train"],
        batch_size=per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    if task_name == "mnli":
        eval_dl_matched = DataLoader(ds["validation_matched"], batch_size=per_device_batch_size, collate_fn=collator)
        eval_dl_mismatched = DataLoader(ds["validation_mismatched"], batch_size=per_device_batch_size, collate_fn=collator)
    else:
        eval_dl = DataLoader(ds[GLUE_EVAL_SPLITS[task_name]], batch_size=per_device_batch_size, collate_fn=collator)

    # Model initialisation
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=info["num_labels"]
    ).to(device)
    
    effective_batch = per_device_batch_size * grad_accum_steps
    if lr_scaling_rule == "linear":
        learning_rate = base_learning_rate * (effective_batch / 32)
    else:
        learning_rate = base_learning_rate
        
    if 'weight_decay' not in optim_params:
        raise ValueError(
            "Optimizer parameters must include 'weight_decay' for GLUE fine-tuning."
        )

    # Optimizer initialization
    optim = optimizer_cls(
        model.parameters(), lr=learning_rate, **optim_params
    )

    # Scheduler
    num_update_steps_per_epoch = len(train_dl) // grad_accum_steps
    max_train_steps = num_update_steps_per_epoch * num_epochs
    warmup_steps = int(warmup_ratio * max_train_steps)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    scaler = GradScaler()
    logs: list[dict[str, float]] = []

    # Training loop
    global_step = 0
    for ep in tqdm(range(1, num_epochs + 1), desc="Epochs", unit="ép"):
        model.train()
        tot_loss, tot_samples = 0.0, 0

        for step, batch in enumerate(
            tqdm(train_dl, desc=f"Train | ép {ep}", leave=False, unit="batch")
        ):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast(device_type="cuda"):
                loss = model(**batch).loss / grad_accum_steps 

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_dl):
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

                lr_scheduler.step()
                global_step += 1

            # Stats
            bsz = batch["labels"].size(0)
            tot_loss += loss.item() * grad_accum_steps * bsz
            tot_samples += bsz

        tr_loss = tot_loss / tot_samples

        # Evaluation
        if task_name == "mnli":
            loss_m, acc_m, _, raw_m = evaluate_model(model, eval_dl_matched, device, task_name, info["is_regression"])
            loss_mm, acc_mm, _, raw_mm = evaluate_model(model, eval_dl_mismatched, device, task_name, info["is_regression"])
            ev_loss = (loss_m + loss_mm) / 2.0
            ev_metric = (acc_m + acc_mm) / 2.0
            metric_name = "Accuracy-Avg(m/mm)"
            raw_metrics = {"matched": raw_m, "mismatched": raw_mm}
            tqdm.write(f"Epoch {ep}/{num_epochs} - Train {tr_loss:.4f} | Eval {ev_loss:.4f} | {metric_name} {ev_metric:.4f} (m: {acc_m:.4f}, mm: {acc_mm:.4f})")
        else:
            ev_loss, ev_metric, metric_name, raw_metrics = evaluate_model(model, eval_dl, device, task_name, info["is_regression"])
            tqdm.write(f"Epoch {ep}/{num_epochs} - Train {tr_loss:.4f} | Eval {ev_loss:.4f} | {metric_name} {ev_metric:.4f}")

        logs.append({
            "epoch": ep,
            "train_loss": tr_loss,
            "eval_loss": ev_loss,
            "aggregated_metric": {metric_name: float(ev_metric)},
            "raw_metrics": raw_metrics
        })
        

    # Save logs
    output_dir = f'logs/GLUE/{task_name}'
    save_logs(
        optimizer_name,
        seed_val,
        logs,
        folder=output_dir,
    )

if __name__ == "__main__":
    import torch
    from optimizers import *
    
    optimizers_to_run = [
            (Adan, 'AdanTuned', {'lr': 7e-5, 'betas': (0.95, 0.92, 0.96), 'weight_decay': 1e-2}),
            (torch.optim.AdamW, 'AdamTuned', {'lr': 7e-5, 'betas': (0.9, 0.99), 'weight_decay': 1e-2}),
            (Ano, 'AnoTuned', {'lr': 2e-5, 'betas': (0.9, 0.999), 'weight_decay': 1e-2}),
            #(Adan, 'AdanBaseline', {'lr': 2e-5, 'weight_decay': 1e-2}),
            #(torch.optim.AdamW, 'AdamBaseline', {'lr': 2e-5, 'weight_decay': 1e-2}),
            #(Ano, 'AnoBaseline', {'lr': 2e-5, 'weight_decay': 1e-2}),
            #(Lion, 'LionBaseline', {'lr': 7e-6, 'weight_decay': 1e-2}),
            #(Grams, 'GramsBaseline', {'lr': 2e-5, 'weight_decay': 1e-2}),
            (Grams, 'GramsTuned', {'lr': 7e-5, 'betas': (0.9, 0.999),'weight_decay': 1e-2}),

            (Anolog, 'Anolog', {'lr': 2e-5, 'weight_decay': 1e-2}),
            (Lion, 'LionTuned', {'lr': 7e-6, 'betas': (0.9, 0.999), 'weight_decay': 1e-2}),
        ]

    seeds_to_run = [2, 5, 10, 42, 444]
    for seed in seeds_to_run:
        #TASKS = ['stsb', 'cola', 'rte', 'mrpc']
        TASKS = ['mrpc']
        #TASKS = ['cola']

        for task in TASKS:
            for opt_class, opt_name, opt_params in optimizers_to_run:
            
                finetune_glue_task(
                    num_epochs=5,
                    task_name=task,
                    seed_val=seed,
                    optimizer_cls=opt_class,
                    optimizer_name=opt_name,
                    optimizer_params=opt_params,
                    use_small_subset=False,
                )
                
                
                
                