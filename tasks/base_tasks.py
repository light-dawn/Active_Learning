import os
import torch
import random
import numpy as np
from tqdm import tqdm
from utils.data import DataUtils
from utils.dataset import dataset_utils
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import SubsetRandomSampler, DataLoader
from metrics import *


class BaseTask:
    def __init__(self, task_name, model, dataset):
        self.task_name = task_name
        self.model = model
        self.dataset = dataset

    # 完整执行整个task的方法
    def run():
        raise NotImplementedError

    # 任务结束时收尾
    def end():
        raise NotImplementedError


class DeepTask(BaseTask):
    def __init__(self, task_name, model, dataset, optimizer, criterion, epochs, batch_size):
        super().__init__(task_name, model, dataset)
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.writer = SummaryWriter(os.path.join("runs", self.task_name))
        self.cur_epoch = 1
        
    def train_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0
        print("Length of the TrainLoader: ", len(train_loader))
        process = tqdm(train_loader, leave=True)
        # Assume that the __getitem__ method in Dataset returns (data, targets, indices)
        for data, targets, _ in process:
            data = data.to(device=self.device, dtype=torch.float32)
            targets = targets.to(device=self.device, dtype=torch.long)
            assert data.shape[1] == self.model.n_channels, "数据通道数与网络通道数不匹配"

            # 预测并计算loss
            prediction = self.model(data)
            loss = self.criterion(prediction, targets)
            epoch_loss += loss.item()

            # 消除上一次梯度，然后反向传播并更新权重
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            process.set_description(f"Train epoch: {self.cur_epoch}, Loss: {loss.item():.5f}")
        return epoch_loss

    def train(self, train_loader):
        for epoch in range(self.epochs):
            epoch_loss = self.train_epoch(train_loader)
            print(f"Epoch: {self.cur_epoch}, Loss: {epoch_loss}")
            self.cur_epoch += 1

    def eval_seg(self, test_loader):
        total = 0
        pa, mpa, miou, fwiou = 0.0, 0.0, 0.0, 0.0
        self.model.eval()
        process = tqdm(test_loader, leave=True)
        for data, targets, _ in process:
            data = data.to(device=self.device, dtype=torch.float32)
            targets = targets.to(device=self.device, dtype=torch.long)
            assert data.shape[1] == self.model.n_channels, "数据通道数与网络通道数不匹配"

            prediction = self.model(data)
            for pred, target in zip(prediction, targets):
                total += 1
                pa += pixel_accuracy(pred, target)
                mpa += mean_accuracy(pred, target)
                miou += mean_IU(pred, target)
                fwiou += frequency_weighted_IU(pred, target)
        if total:
            pa, mpa, miou, fwiou = pa / total, mpa / total, miou / total, fwiou / total
        else:
            raise Exception("total number of data is 0")
        return {"pixel_acc": pa, "mean_pixel_acc": mpa, "mean_iou": miou, "frequency_weighted_iou": fwiou}

    def run(self):
        for epoch in range(self.epochs):
            epoch_loss = self.train_epoch(train_loader)
            print(f"Epoch: {self.cur_epoch}, Loss: {epoch_loss}")
            self.cur_epoch += 1

    def end(self):
        self.writer.close()

    # 测试方法依赖于不同任务的评估方式，放到子类实现
    def validation(self, test_loader):
        raise NotImplementedError


class DeepActiveTask(DeepTask):
    def __init__(self, task_name, model, dataset, optimizer, criterion, epochs, batch_size, 
                 init_budget, budget, cycles):
        super().__init__(task_name, model, dataset, optimizer, criterion, epochs, batch_size)
        self.budget = budget
        self.cycles = cycles
        self.unlabeled_indices, self.labeled_indices = self._init_labeling(init_budget)
        # print("Labeled dataset initialized.")
        # print("Current labeled number: ", len(self.labeled_indices))
        # print("Current unlabeled number: ", len(self.unlabeled_indices))
        # print("Unlabeled indices: ", self.unlabeled_indices)

    def _init_labeling(self, init_budget):
        unlabeled_indices = list(range(len(self.dataset)))
        # 初始化labeled数据
        labeled_indices = random.sample(unlabeled_indices, init_budget)
        # 将请求到标签的数据从无标签数据池中移除
        unlabeled_indices = np.setdiff1d(unlabeled_indices, labeled_indices)
        return unlabeled_indices, labeled_indices

    def query_and_move(self, queried_indices):
        self.labeled_indices.extend(queried_indices)
        # print("Labeled indices: ", self.labeled_indices)
        # print("Unlabeled indices: ", self.unlabeled_indices)
        count = 0
        for i in queried_indices:
            if i not in self.unlabeled_indices:
                print(f"{i} not in unlabeled_indices")
                count += 1
        # print("Not in count: ", count)
        self.unlabeled_indices = np.setdiff1d(self.unlabeled_indices, queried_indices)
        self.cur_epoch = 1

    def get_cur_data_loader(self, part="labeled"):
        assert part in ("labeled", "unlabeled"), "parameter 'part' must be 'labeled' or 'unlabeled'"
        if part == "labeled":
            sampler = SubsetRandomSampler(self.labeled_indices)
        else:
            sampler = SubsetRandomSampler(self.unlabeled_indices)
        return DataLoader(self.dataset, batch_size=self.batch_size, sampler=sampler, pin_memory=False)
        
        

if __name__ == "__main__":
    task = ActiveLearningTask("test", task_type="seg", data_root=config["data_root"]["segmentation"])
    print(type(task.pool))
    print(len(task.pool))
        
        