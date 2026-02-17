# scripts/train.py
import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler

# Your src imports — adjust if paths differ
from src.utils.seed import set_all_seeds
from src.utils.logging import setup_logging
from src.utils.unfreeze import unfreeze_last_swin_stages
from src.data.dataset import CUBVisionDataset   # assuming you fixed globals/paths
from src.model.vitmix_swin import ViTMixSwin
from src.engine.train import train_model
from torch.utils.data import DataLoader, Subset
import random
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/swin_b.yaml")
    return parser.parse_args()

def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Merge base if needed (simple override for now)
    base_path = Path("configs/base.yaml")
    if base_path.exists():
        with open(base_path) as f:
            base = yaml.safe_load(f)
        cfg = {**base, **cfg}  # simple dict merge

    # Setup
    set_all_seeds(cfg["seed"])
    setup_logging(log_file=Path(cfg["paths"]["save_dir"]) / cfg["logging"]["log_file"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data root from config (no more /kaggle/ hardcode)
    data_root = cfg["paths"]["data_root"]
    metadata_dir = cfg["paths"]["metadata_dir"]

    # You need to implement get_split_ids or load train/test ids here
    # For now — placeholder (replace with your stratified split logic)
    # Example:
    # train_ids, test_ids = load_cub_splits(data_root)
    # Then create datasets/loaders using cfg["batch_size"], etc.

    # Model
    model = ViTMixSwin(freeze_vision=cfg["model"]["freeze_vision"]).to(device)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["T_max"])
    criterion = nn.CrossEntropyLoss()

    # Training call (your train_model from engine)
    train_model(
        model=model,
        train_loader=... ,  # ← fill this
        val_loader=... ,    # ← fill this
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=cfg["epochs"],
        scheduler=scheduler,
        amp=cfg["amp"],
        unfreeze_after=cfg["unfreeze_after"],
        save_dir=cfg["paths"]["save_dir"],
        best_model_name=cfg["logging"]["best_model_name"],
        log_csv_name=cfg["logging"]["csv_name"]
    )

    print("Training done. Best model saved.")

if __name__ == "__main__":
    main()