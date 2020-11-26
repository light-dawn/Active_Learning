import os
import torch
import random
import numpy as np
from tqdm import tqdm
from utils.data import DataUtils
from config import config
from utils.dataset import dataset_utils
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import SubsetRandomSampler


# 任务类型映射到方法名称，用于反射调用
dataset_map = { # 构造数据集
    "cla": "create_image_cla_dataset",
    "seg": "create_image_seg_dataset"
}
data_load_map = { # 加载数据路径
    "cla": "load_labeled_data_paths",
    "seg": "load_seg_data_paths"
}
transform_map = { # 预处理变换
    "cla": {"x": "image_resize", "y": None},
    "seg": {"x": "image_resize", "y": "process_masks"}
}


class BaseTask:
    def __init__(self, task_name, model, dataset):
        self.task_name = task_name
        self.model = model
        self.dataset = dataset

    # 完整执行整个task的方法
    def run():
        raise NotImplementedError


class DeepTask(BaseTask):
    def __init__(self, task_name, model, dataset, optimizer, criterion, epochs, batch_size):
        super().__init__(task_name, model, dataset)
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(os.path.join(config["tensorboard_log_dir"], self.task_name))
        self.cur_epoch = 1

    def train_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0
        process = tqdm(train_loader, leave=True)
        for data, targets in process:
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
            process.set_description(f"Train epoch: {cur_epoch + 1}, Loss: {loss.item()}")
   
        return epoch_loss

    def train(self, train_loader):
        for epoch in range(self.epochs):
            epoch_loss = self.train_epoch(train_loader)
            print(f"Epoch: {self.cur_epoch}, Loss: {epoch_loss}")
            self.cur_epoch += 1

    def run(self):
        self.train()

    # 测试方法依赖于不同任务的评估方式，放到子类实现
    def validation(self, test_loader):
        raise NotImplementedError


class ActiveTask(BaseTask):
    def __init__(self, task_name, model, dataset, init_budget, budget, cycles):
        super().__init__(task_name, model, dataset)
        self.budget = budget
        self.cycles = cycles
        self.unlabeled_indices, self.labeled_indices = self._init_labeling(init_budget)

    def _init_labeling(self, init_budget):
        unlabeled_indices = list(range(len(self.dataset)))
        # 初始化labeled数据
        labeled_indices = random.sample(unlabeled_indices, init_budget)
        # 将请求到标签的数据从无标签数据池中移除
        unlabeled_indices = np.setdiff1d(unlabeled_indices, labeled_indices)
        return unlabeled_indices, labeled_indices

    def query_and_move(self, queried_indices):
        self.labeled_indices.extend(queried_indices)
        self.unlabeled_indices = np.setdiff1d(self.unlabeled_indices, queried_indices)
        

class DeepClaTask(DeepTask):
    def __init__(self, task_name, model, dataset, optimizer, criterion, epochs, batch_size):
        super().__init__(task_name, model, dataset, optimizer, criterion, epochs, batch_size)

    def test(self, test_loader):
        self.model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for data, targets in test_loader:
                data = data.to(devices)
                targets = targets.to(devices)

                scores, _ = self.model(data)
                _, preds = torch.max(scores.data, 1)
                total += targets.size(0)
                correct += (preds == targets).sum().item()
        return 100 * correct / total


class DeepSegTask(DeepTask):
    def __init__(self, task_name, model, dataset, optimizer, criterion, epochs, batch_size):
        super().__init__(task_name, model, dataset, optimizer, criterion, epochs, batch_size)

    def test(self, test_loader):
        raise NotImplementedError


class DeepActiveTask(DeepTask, ActiveTask):
    def __init__(self, task_name, model, dataset, optimizer, criterion, epochs, batch_size, 
                 init_budget, budget, cycles):
        DeepTask.__init__(task_name, model, dataset, optimizer, criterion, epochs, batch_size)
        ActiveTask.__init__(task_name, model, dataset, init_budget, budget, cycles)

    def get_cur_train_loader(self):
        sampler = SubsetRandomSampler(self.labeled_indices)
        return DataLoader(self.dataset, batch_size=self.batch_size, sampler=sampler, pin_memory=False)

    
class DeepActiveSegTask(DeepSegTask, DeepActiveTask):
    def __init__(self, task_name, model, dataset, optimizer, criterion, epochs, batch_size, 
                 init_budget, budget, cycles):
        DeepSegTask.__init__(task_name, model, dataset, optimizer, criterion, epochs, batch_size)
        DeepActiveTask.__init__(task_name, model, dataset, optimizer, criterion, epochs, batch_size, 
                                init_budget, budget, cycles)


class DeepActiveClaTask(DeepClaTask, DeepActiveTask):
    def __init__(self, task_name, model, dataset, optimizer, criterion, epochs, batch_size, 
                 init_budget, budget, cycles):
        DeepClaTask.__init__(task_name, model, dataset, optimizer, criterion, epochs, batch_size)
        DeepActiveTask.__init__(task_name, model, dataset, optimizer, criterion, epochs, batch_size, 
                                init_budget, budget, cycles)
        

if __name__ == "__main__":
    task = ActiveLearningTask("test", task_type="seg", data_root=config["data_root"]["segmentation"])
    print(type(task.pool))
    print(len(task.pool))
        
        