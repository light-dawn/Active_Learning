import os
import torch
import random
import numpy as np
import datetime
import json
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import SubsetRandomSampler, DataLoader
from utils.data import *
from utils.utils import modelUtils, lossUtils, optimUtils
from utils.metrics import *


class BaseTask:
    def __init__(self, dataset, conf):
        self.task_name = conf["task"]["name"]
        self.model = getattr(modelUtils, conf["model"]["name"])(conf["model"])
        self.dataset = dataset
        self.conf = conf
        if self.task_name:
            print(f"Task: {self.task_name}")
    
    # Do something at the start of the task, such as printing some task information
    def start(self):
        raise NotImplementedError

    # 完整执行整个task的方法
    def run(self):
        raise NotImplementedError

    def run_one_step(self):
        raise NotImplementedError

    # 任务结束时收尾
    def end(self):
        raise NotImplementedError


class DeepTask(BaseTask):
    def __init__(self, dataset, conf):
        super().__init__(dataset, conf)
        self.optimizer = getattr(optimUtils, conf["optimizer"]["name"])(self.model.parameters(), conf["optimizer"])
        self.criterion = getattr(lossUtils, conf["train"]["criterion"])()
        self.epochs = conf["train"]["epochs"]
        self.batch_size = conf["train"]["batch_size"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.writer = SummaryWriter(os.path.join("runs", self.task_name+"-"+
                      datetime.datetime.strftime(datetime.datetime.now(), 
                      "%Y-%m-%d-%H-%M-%S"))) if conf["train"]["write_tensorboard"] else None
    
    def start(self):
        seed = self.conf["train"]["seed"]
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("Model: ", self.conf["model"]["name"])
        print("Epochs: ", self.epochs)

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
            process.set_description(f"Loss: {loss.item():.5f}")
        return epoch_loss

    def train(self, train_loader):
        train_loss = 0
        for epoch in range(self.epochs):
            epoch_loss = self.train_epoch(train_loader)
            print(f"Epoch: {epoch + 1}, Loss: {epoch_loss}")
            # self.save_model_state_dict(os.path.join("checkpoints", self.task_name+"_Epoch_"+str(self.cur_epoch)+".pth"))
            train_loss += epoch_loss
        return train_loss
            
    def eval_seg(self, test_loader):
        total = 0
        pa, mpa, miou, fwiou = 0.0, 0.0, 0.0, 0.0
        self.model.eval()
        process = tqdm(test_loader, leave=True)
        with torch.no_grad():
            for data, targets, _ in process:
                data = data.to(device=self.device, dtype=torch.float32)
                targets = targets.to(device=self.device, dtype=torch.long)
                assert data.shape[1] == self.model.n_channels, "数据通道数与网络通道数不匹配"
                if self.conf["model"]["name"].endswith("feat"):
                    prediction, _ = self.model(data)
                else:
                    prediction = self.model(data)
                for pred, target in zip(prediction, targets):
                    target = target.cpu().numpy()
                    pred = pred.cpu().numpy()
                    pred = seg_pred_to_mask(pred)
                    total += 1
                    pa += pixel_accuracy(pred, target)
                    mpa += mean_accuracy(pred, target)
                    miou += mean_IU(pred, target)
                    fwiou += frequency_weighted_IU(pred, target)
            print({"pixel_acc": pa, "mean_pixel_acc": mpa, "mean_iou": miou, "frequency_weighted_iou": fwiou})
        if total:
            pa, mpa, miou, fwiou = pa / total, mpa / total, miou / total, fwiou / total
        else:
            raise Exception("total number of data is 0")
        return {"pixel_acc": pa, "mean_pixel_acc": mpa, "mean_iou": miou, "frequency_weighted_iou": fwiou}

    def eval_cla(self, test_loader):
        raise NotImplementedError

    def infer(self, data_dir):
        self.model.eval()
        data = load_and_preprocess_single_image(data_dir)
        with torch.no_grad():
            pred = self.model(data)
        mask = create_visual_anno(pred)
        return mask

    def save_model_state_dict(self, save_dir):
        torch.save(self.model.state_dict(), save_dir)
    
    def load_model_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def run(self):
        for epoch in range(self.epochs):
            epoch_loss = self.train_epoch(train_loader)
            print(f"Epoch: {epoch + 1}, Loss: {epoch_loss}")

    def end(self):
        if self.writer:
            self.writer.close()

    # 测试方法依赖于不同任务的评估方式，放到子类实现
    def validation(self, test_loader):
        raise NotImplementedError


class DeepActiveTask(DeepTask):
    def __init__(self, dataset, conf, unlabeled_indices=None):
        super().__init__(dataset, conf)
        self.budget = conf["active"]["budget"]
        self.cycles = conf["active"]["cycles"]
        self.train_indices, self.test_indices = indices_train_test_split(list(range(len(self.dataset))))
        unlabeled_indices = unlabeled_indices if unlabeled_indices else self.train_indices
        self.unlabeled_indices, self.labeled_indices = self._init_labeling(conf["active"]["init_budget"], unlabeled_indices)
        # print("Labeled dataset initialized.")
        # print("Current labeled number: ", len(self.labeled_indices))
        # print("Current unlabeled number: ", len(self.unlabeled_indices))
        # print("Unlabeled indices: ", self.unlabeled_indices)
    
    def start(self):
        super().start()
        print("Initial Budget: ", self.conf["active"]["init_budget"])
        print("Cycle Budget: ", self.budget)
        print("Cycles: ", self.cycles)
        print("Task Start! \n ----------------------")

    def run(self):
        test_sampler = SubsetRandomSampler(self.test_indices)
        test_loader = DataLoader(self.dataset, batch_size=1, sampler=test_sampler)
        for cycle in range(self.cycles):
            print("Cycle: ", cycle + 1)
            if len(self.unlabeled_indices) == 0:
                print("All data samples are labeled. Task finish.")
                break
            cycle_loss = self.run_one_step()
            if self.writer:
                self.writer.add_scalar("Loss/train", cycle_loss, cycle + 1)
            print("Start Evaluation.")
            metrics = self.eval_seg(test_loader) if self.conf["task"]["type"] == "seg" else self.eval_cla(test_loader)
            for metric_name, metric_value in metrics.items():
                print(f"{metric_name}: {metric_value}")
                if self.writer:
                    self.writer.add_scalar(metric_name+"/eval", metric_value, cycle + 1)
            print("Query and move data...\n")

    def _init_labeling(self, init_budget, unlabeled_indices):
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
        
        