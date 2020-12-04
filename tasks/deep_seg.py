from tasks.base_tasks import DeepTask
from models import unet
import json
from utils.data import DataUtils
from utils.dataset import ImageSegDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import nn, optim
import os
import random


class DeepSegPipeline(DeepTask):
    def __init__(self, task_name, model, dataset, optimizer, criterion, epochs, batch_size):
        super().__init__(task_name, model, dataset, optimizer, criterion, epochs, batch_size)
    
    def get_train_test_loader(self, test_size=0.3, seed=2020):
        random.seed(seed)
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        # print(indices)
        train_indices, test_indices = indices[:int((1-test_size)*len(indices))], indices[int((1-test_size)*len(indices)):]
        train_sampler, test_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(test_indices)
        train_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=train_sampler)
        test_loader = DataLoader(self.dataset, batch_size=1, sampler=test_sampler)
        return train_loader, test_loader

    def run(self):
        train_loader, test_loader = self.get_train_test_loader()
        for _ in range(self.epochs):
            epoch_loss = self.train_epoch(train_loader)
            self.writer.add_scalar("Loss/train", epoch_loss, self.cur_epoch)
            print(f"Epoch: {self.cur_epoch}")
            print(f"Loss: {epoch_loss}")
            metrics = self.eval_seg(test_loader)
            for metric_name, metric_value in metrics.items():
                print(f"{metric_name}: {metric_value}")
                self.writer.add_scalar(metric_name+"/eval", metric_value, self.cur_epoch)
            self.cur_epoch += 1


def pipeline(config):
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
    epochs, batch_size = config["train"]["epochs"], config["train"]["batch_size"]
    # 任务
    task = DeepSegPipeline("deep_seg_trail", model, dataset, optimizer, criterion, epochs, batch_size)
    task.run()
    task.end()


if __name__ == "__main__":
    pipeline()