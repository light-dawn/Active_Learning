# encoding=utf-8
from models import unet
from tasks.random_samping import RandomSegPipeline
import json
 
from utils.data import DataUtils
from utils.dataset import ImageSegDataset
import random
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import nn, optim
import os

path = os.getcwd()
print("Path: ", path)

# 之后调整为读取控制台参数
with open("demo_cfg.json", "r") as f:
    config = json.loads(f.read())
print(config)


if __name__ == "__main__":
    # 模型
    model = unet.UNet(n_channels=config["model"]["n_channels"], n_classes=config["model"]["n_classes"])
    # 数据集
    data_paths, mask_paths = DataUtils.load_seg_data_paths(config["data"]["data_root"], sep="\\")
    print("Data path numbers: ", len(data_paths))
    dataset = ImageSegDataset(data_paths, mask_paths, DataUtils.image_resize, DataUtils.process_masks)
    print("Dataset size: ", len(dataset))
    # 优化器
    optimizer = optim.RMSprop(model.parameters(), lr=config["optimizer"]["lr"], 
                              weight_decay=config["optimizer"]["weight_decay"], 
                              momentum=config["optimizer"]["momentum"])
    # 损失函数
    if model.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    # 初始标注预算
    init_budget, budget, cycles = config["active"]["init_budget"], config["active"]["budget"], config["active"]["cycles"]
    epochs, batch_size = config["train"]["epochs"], config["train"]["batch_size"]
    # 任务
    task = RandomSegPipeline("active_seg_demo", model, dataset, optimizer, criterion, epochs, batch_size, init_budget, budget, cycles)
    task.run()

