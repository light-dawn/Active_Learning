import sys
sys.path.append("./")
import torch
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from models import unet

lr = 1e-3
batch_size = 12
with SummaryWriter("runs/demo") as w:
    x = torch.zeros((1, 3, 224, 224))
    w.add_graph(unet.UNet(n_channels=3, n_classes=3), x)

# for i in range(10):
#     writer.add_scalar("quadratic", i ** 2, global_step=i)
#     writer.add_scalar("exponential", 2 ** i, global_step=i)

# writer.flush()