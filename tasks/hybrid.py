from tasks.base_tasks import DeepActiveTask
from sampler import HybridSampler

from models import unet
import json

from utils.data import DataUtils
from utils.dataset import ImageSegDataset
import random
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import nn, optim
import os


class HybridPipeline(DeepActiveTask):
    def __init__(self, task_name, model, dataset, optimizer, criterion, epochs, batch_size, 
                 init_budget, budget, cycles, lossnet_feature_sizes, lossnet_num_channels):
        super().__init__(task_name, model, dataset, optimizer, criterion, epochs, batch_size, 
                         init_budget, budget, cycles)
        self.lossnet = lossnet.LossNet(feature_sizes=lossnet_feature_sizes, num_channels=lossnet_num_channels)
        self.sampler = HybridSampler(self.budget, self.lossnet, self.model, self.device)
    
    def run(self):
        for cycle in range(self.cycles):
            print("Cycle: ", cycle + 1)
            # 构造DataLoader
            labeled_loader = self.get_cur_data_loader(part="labeled")
            unlabeled_loader = self.get_cur_data_loader(part="unlabeled")
            self.train(labeled_loader)
            query_indices = self.sampler.sample(labeled_loader, unlabeled_loader)
            self.query_and_move(query_indices)
            print("Query and move data...")


def pipeline(config):
    # 模型
    model = unet.UNet_FM(n_channels=config["model"]["n_channels"], n_classes=config["model"]["n_classes"])
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
    # 损失网络定义
    lossnet_feature_sizes = config["lossnet"]["feature_sizes"]
    lossnet_num_channels = config["lossnet"]["num_channels"]
    # 任务
    task = HybridPipeline("hybrid_trail", model, dataset, optimizer, criterion, epochs, batch_size, 
                          init_budget, budget, cycles, lossnet_feature_sizes, lossnet_num_channels)
    task.run()


if __name__ == "__main__":
    random_sampling()




