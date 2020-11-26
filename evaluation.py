import torch
import torch.nn.functional as F
from tqdm import tqdm

from loss.dice_loss import dice_coeff


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for data, targets in loader:
            data = data.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=mask_type)

            with torch.no_grad():
                prediction = net(data)

            if net.n_classes > 1:
                tot += F.cross_entropy(prediction, targets).item()
            else:
                pred = torch.sigmoid(prediction)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, targets).item()
            pbar.update()

    net.train()
    return tot / n_val


def infer(net, data, device):
    net.eval()
    with torch.no_grad():
        prediction = net(data)
    