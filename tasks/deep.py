import os
import json
import torch
import random
from torch import nn, optim
from torch.utils.data import DataLoader, SubsetRandomSampler

from models import unet
from utils.metrics import metricUtils
from utils.dataset import ImageSegDataset
from tasks.base_tasks import DeepTask


class DeepPipeline(DeepTask):
    def __init__(self, dataset, conf):
        super().__init__(dataset, conf)
    
    def get_train_test_loader(self, test_size=0.2, seed=2020):
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
        for epoch in range(self.epochs):
            epoch_loss = self.train_epoch(train_loader)
            if self.writer:
                self.writer.add_scalar("Loss/train", epoch_loss, epoch + 1)
            print(f"Epoch: {epoch + 1}")
            print(f"Loss: {epoch_loss}")
            metrics_func_dict = getattr(metricUtils, self.conf["task"]["type"])()
            metrics = self.eval_seg(test_loader, metrics_func_dict) if self.conf["task"]["type"] == "seg" else self.eval_cla(test_loader)
            for metric_name, metric_value in metrics.items():
                print(f"{metric_name}: {metric_value}")
                if metric_name not in self.train_status:
                    self.train_status[metric_name] = [metric_value]
                else:
                    self.train_status[metric_name].append(metric_value)
                if self.writer:
                    self.writer.add_scalar(metric_name+"/eval", metric_value, epoch + 1)
            if self.conf["train"]["save_best_model"]:   
                best_count = 0
                for metric_name, value_list in self.train_status.items():
                    if metrics[metric_name] == max(value_list):
                        best_count += 1
                if best_count == len(metrics):
                    print("Save the best model.")
                    torch.save(self.model.state_dict(), os.path.join("checkpoints", self.conf["task"]["name"]+".pth"))
                    
            # if self.save_weight:
            #     self.save_model_state_dict(os.path.join("checkpoints", self.task_name+"_Epoch_"+str(epoch + 1)+".pth"))
