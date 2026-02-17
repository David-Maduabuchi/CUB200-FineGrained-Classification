import torch
from tqdm.auto import tqdm

from .metrics import accuracy, AverageMeter


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter("val_loss")
    acc1_meter = AverageMeter("val_acc1")
    acc5_meter = AverageMeter("val_acc5")

    for images, labels in tqdm(loader, desc="Validate", leave=False, colour="red"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        acc1, acc5 = accuracy(outputs, labels, topk=(1,5))
        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        acc1_meter.update(acc1.item(), bs)
        acc5_meter.update(acc5.item(), bs)

    return {
        "val_loss": loss_meter.avg,
        "val_acc1": acc1_meter.avg,
        "val_acc5": acc5_meter.avg
    }