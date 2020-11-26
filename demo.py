from models import unet
from tasks.random_samping import RandomSegPipeline
import json

import utils.data as data
from utils.dataset import ImageSegDataset
import random
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import nn

# 之后调整为读取控制台参数
config = json.loads("demo_cfg.json")


if __name__ == "__main__":
    # 模型
    model = unet.UNet(n_channels=3, n_classes=4)
    # 数据集
    data_paths, mask_paths = data.load_seg_data_paths(config["data"]["data_root"])
    dataset = ImageSegDataset(data_paths, mask_paths, data.image_resize, data.process_masks)
    # 优化器
    optimizer = optim.RMSprop(model.parameters(), lr=config["optimizer"]["lr"], 
                              weight_decay=config["optimizer"]["weight_decay"], 
                              momentum=config["optimizer"]["momentum"])
    # 损失函数
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    # 初始标注预算
    init_budget, budget, cycles = config["active"]["init_budget"], config["active"]["budget"], config["active"]["cycles"]   
    # 任务
    task = RandomSegPipeline("active_seg_demo", model, dataset, optimizer, criterion, init_budget, budget, cycles)
    task.run()

