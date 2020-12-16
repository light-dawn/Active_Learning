import os
import json
import torch
import random
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader, SubsetRandomSampler

from models import unet, lossnet
from loss import loss_prediction_loss
from sampler import *
from utils.dataset import ImageSegDataset
from utils.utils import optimUtils
from tasks.base_tasks import DeepActiveTask


class LossPredPipeline(DeepActiveTask):
    def __init__(self, dataset, conf, unlabeled_indices=None):
        super().__init__(dataset, conf, unlabeled_indices)
        self.lossnet = lossnet.LossNet(feature_sizes=conf["lossnet"]["feature_sizes"], 
                                       num_channels=conf["lossnet"]["num_channels"])
        self.lossnet_optimizer = getattr(optimUtils, self.conf["lossnet"]["optimizer"]["name"])(self.lossnet.parameters(), conf["lossnet"]["optimizer"])
        self.lossnet.to(self.device)
        self.sampler = LossPredictionSampler(self.budget, self.model, self.lossnet, self.device)
    
    # Jointly train the lossnet along with the task model
    def train_epoch(self, train_loader):
        self.model.train()
        self.lossnet.train()
        epoch_loss, epoch_loss_pred_loss = 0.0, 0.0
        print("Length of the TrainLoader: ", len(train_loader))
        process = tqdm(train_loader, leave=True)
        # Assume that the __getitem__ method in Dataset returns (data, targets, indices)
        for data, targets, _ in process:
            data = data.to(device=self.device, dtype=torch.float32)
            targets = targets.to(device=self.device, dtype=torch.long)
            assert data.shape[1] == self.model.n_channels, "数据通道数与网络通道数不匹配"

            # predict and get feature maps
            prediction, feature_maps = self.model(data)
            task_loss = self.criterion(prediction, targets)
            if len(task_loss.size()) >= 3:
                task_loss = task_loss.mean(dim=(1, 2)).unsqueeze(1)
            epoch_loss += task_loss.mean().item()
            feature_maps = [feat.detach() for feat in feature_maps]
            pred_loss = self.lossnet(feature_maps)
            pred_loss.view(pred_loss.size(0))
            loss_pred_loss = loss_prediction_loss(pred_loss, task_loss)
            epoch_loss_pred_loss += loss_pred_loss

            self.optimizer.zero_grad()
            self.lossnet_optimizer.zero_grad()
            task_loss.mean().backward()
            loss_pred_loss.backward()
            self.optimizer.step()
            self.lossnet_optimizer.step()

            process.set_description(f"Task Loss: {task_loss.mean().item():.5f}, Loss Pred Loss: {loss_pred_loss}")
        print("Mean Epoch Loss Pred Loss: ", epoch_loss_pred_loss.item() / len(process))
        return epoch_loss

    def run_one_step(self):
        # 构造DataLoader
        labeled_loader, unlabeled_loader = self.get_cur_data_loader(part="labeled"), self.get_cur_data_loader(part="unlabeled")
        train_loss = self.train(labeled_loader)
        query_indices = self.sampler.sample(unlabeled_loader)
        self.query_and_move(query_indices)
        return train_loss / len(self.labeled_indices)


class HybridPipeline(LossPredPipeline):
    def __init__(self, dataset, conf, unlabeled_indices=None):
        super().__init__(dataset, conf, unlabeled_indices)
        self.sampler = HybridSampler(self.budget, self.model, self.lossnet, self.device)

    def run_one_step(self):
        # 构造DataLoader
        labeled_loader, unlabeled_loader = self.get_cur_data_loader(part="labeled"), self.get_cur_data_loader(part="unlabeled")
        train_loss = self.train(labeled_loader)
        query_indices = self.sampler.sample(labeled_loader, unlabeled_loader)
        self.query_and_move(query_indices)
        return train_loss / len(self.labeled_indices)


class RandomPipeline(DeepActiveTask):
    def __init__(self, dataset, conf, unlabeled_indices=None):
        super().__init__(dataset, conf, unlabeled_indices)
        self.sampler = RandomSampler(self.budget)

    def run_one_step(self):
        # 构造DataLoader
        labeled_loader, unlabeled_loader = self.get_cur_data_loader(part="labeled"), self.get_cur_data_loader(part="unlabeled")
        train_loss = self.train(labeled_loader)
        query_indices = self.sampler.sample(unlabeled_loader)
        self.query_and_move(query_indices)
        return train_loss / len(self.labeled_indices)


class EmbDistPipeline(DeepActiveTask):
    def __init__(self, dataset, conf, unlabeled_indices=None):
        super().__init__(dataset, conf, unlabeled_indices)
        self.sampler = EmbDistSampler(self.budget, self.model, self.device)

    def run_one_step(self):
        # 构造DataLoader
        labeled_loader, unlabeled_loader = self.get_cur_data_loader(part="labeled"), self.get_cur_data_loader(part="unlabeled")
        train_loss = self.train(labeled_loader)
        query_indices = self.sampler.sample(labeled_loader, unlabeled_loader)
        self.query_and_move(query_indices)
        return train_loss / len(self.labeled_indices)


class ActiveUtils:
    @staticmethod
    def hybrid(dataset, conf, unlabeled_indices=None):
        return HybridPipeline(dataset, conf, unlabeled_indices)

    @staticmethod
    def random(dataset, conf, unlabeled_indices=None):
        return RandomPipeline(dataset, conf, unlabeled_indices)

    @staticmethod
    def loss_pred(dataset, conf, unlabeled_indices=None):
        return LossPredPipeline(dataset, conf, unlabeled_indices)

    @staticmethod
    def emb_dist(dataset, conf, unlabeled_indices=None):
        return EmbDistPipeline(dataset, conf, unlabeled_indices)


activeUtils = ActiveUtils()











