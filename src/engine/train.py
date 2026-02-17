import os
import torch
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

from .metrics import accuracy, AverageMeter
from .validate import validate_one_epoch


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    loss_meter = AverageMeter("train_loss")
    acc1_meter = AverageMeter("train_acc1")

    for images, labels in tqdm(loader, desc="Train", leave=False, colour="green"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        acc1 = accuracy(outputs, labels, topk=(1,))[0]
        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        acc1_meter.update(acc1.item(), bs)

    return {"train_loss": loss_meter.avg, "train_acc1": acc1_meter.avg}


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs=20,
    scheduler=None,
    amp=True,
    unfreeze_after=5,
    save_dir=".",
    best_model_name="best_model.pth",
    log_csv_name="training_log.csv"
):
    os.makedirs(save_dir, exist_ok=True)
    scaler = GradScaler() if amp else None
    best_val_acc = 0.0

    log_path = os.path.join(save_dir, log_csv_name)
    df_columns = ["epoch", "train_loss", "train_acc1", "val_loss", "val_acc1", "val_acc5"]
    pd.DataFrame(columns=df_columns).to_csv(log_path, index=False)

    for epoch in range(1, epochs + 1):
        if epoch == unfreeze_after:
            print(">> Unfreezing last 2 Swin blocks...")
            model.unfreeze_last_stages(num_stages=2)

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_metrics = validate_one_epoch(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step()

        metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["train_loss"],
            "train_acc1": train_metrics["train_acc1"],
            "val_loss": val_metrics["val_loss"],
            "val_acc1": val_metrics["val_acc1"],
            "val_acc5": val_metrics["val_acc5"]
        }

        pd.DataFrame([metrics]).to_csv(log_path, mode='a', header=False, index=False)

        if val_metrics["val_acc1"] > best_val_acc:
            best_val_acc = val_metrics["val_acc1"]
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc1": best_val_acc
            }, os.path.join(save_dir, best_model_name))

        print(f"Epoch [{epoch}/{epochs}] | "
              f"Train Loss: {train_metrics['train_loss']:.4f}, Acc: {train_metrics['train_acc1']:.2f}% | "
              f"Val Loss: {val_metrics['val_loss']:.4f}, Acc: {val_metrics['val_acc1']:.2f}%")

    return None  # you can return history if you want, but csv is saved anyway