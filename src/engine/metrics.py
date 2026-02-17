import torch

@torch.no_grad()
def accuracy(logits, targets, topk=(1,)):
    maxk = max(topk)
    batch_size = targets.size(0)
    _, preds = logits.topk(maxk, dim=1, largest=True, sorted=True)
    preds = preds.t()
    correct = preds.eq(targets.view(1, -1).expand_as(preds))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter:
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = self.sum = self.count = self.avg = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0