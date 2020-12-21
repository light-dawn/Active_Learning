import os
import json
import torch
import random
from copy import deepcopy
from torch import nn, optim
from torch.utils.data import DataLoader, SubsetRandomSampler

from models import unet, discriminator
from sampler import LossPredictionSampler, EmbDistSampler
from tasks.base_tasks import DeepTask
from tasks.active import activeUtils, HybridPipeline
from utils.data import indices_train_test_split
from utils.dataset import ImageSegDataset
from utils.federate import partition, average_weights
from utils.utils import optimUtils
from utils.metrics import metricUtils


# The client nodes of TEFAL Framework need extra capability of loss prediction and embedding distance computing
class TefalClient(HybridPipeline):
    def __init__(self, dataset, conf):
        super().__init__(dataset, conf)

    # total_pred_loss, total_emb_dist
    def client_evaluation(self, dataloader):
        embeddings, all_pred_loss, _ = self.get_feature_embedding_and_pred_loss(dataloader)
        centroid = embeddings.mean(0)
        dists = torch.norm(embeddings - centroid, p=2, dim=1)
        total_pred_loss, total_emb_dist = all_pred_loss.sum().data, dists.sum().data
        return total_pred_loss, emb_dist
    

class FedPipeline(DeepTask):
    def __init__(self, dataset, conf):
        super().__init__(dataset, conf)
        print(self.epochs)
        self.client_nums = self.conf["federate"]["client_nums"]
        self.clients = {}
        for i in range(self.client_nums):
            # the param 'dataset' is useless in clients' tasks
            # self.clients[i] = DeepTask(task_name+"client"+str(i), deepcopy(model), dataset, optimizer, criterion, epochs, batch_size, 
            #                            write_tensorboard=False)
            self.clients[i] = DeepTask(dataset, conf)

    def get_client_train_loader(self, train_indices, seed=2020):
        client_indices_dict = partition(train_indices, self.client_nums)
        client_trainloader_dict = {}
        for client, indices in client_indices_dict.items():
            # print(f"Client: {client}, Data numbers: {len(indices)}")
            sampler = SubsetRandomSampler(indices)
            client_trainloader_dict[client] = DataLoader(self.dataset, batch_size=self.batch_size, sampler=sampler)
        return client_trainloader_dict
        
    def run(self):
        train_indices, test_indices = indices_train_test_split(list(range(len(self.dataset))))
        print("Train data numbers: ", len(train_indices))
        test_sampler = SubsetRandomSampler(test_indices)
        test_loader = DataLoader(self.dataset, batch_size=1, sampler=test_sampler)
        client_train_loaders = self.get_client_train_loader(train_indices)
        # self.model.train()
        for round in range(self.conf["federate"]["round"]):
            print(f"\n Federated Round {round + 1}")
            local_weights, local_loss = [], []
            for i in range(self.client_nums):
                # For each client, load the global weights first
                print(f"--------------------\nClient {i} local training.")
                self.clients[i].model.load_state_dict(deepcopy(self.model.state_dict()))
                self.clients[i].model.to(self.clients[i].device)
                loss = self.clients[i].train(client_train_loaders[i])
                local_loss.append(deepcopy(loss))
                local_weights.append(deepcopy(self.clients[i].model.cpu().state_dict()))
            global_weight = average_weights(local_weights)
            self.model.load_state_dict(global_weight)
            self.model.to(self.device)
            global_loss = sum(local_loss) / len(local_loss)
            print("Train Loss: ", global_loss)
            if self.writer:
                self.writer.add_scalar("Loss/train", global_loss, round + 1)
            metrics_func_dict = getattr(metricUtils, self.conf["task"]["type"])()
            metrics = self.eval_seg(test_loader, metrics_func_dict) if self.conf["task"]["type"] == "seg" else self.eval_cla(test_loader)
            for metric_name, metric_value in metrics.items():
                print(f"{metric_name}: {metric_value}")
                if metric_name not in self.train_status:
                    self.train_status[metric_name] = [metric_value]
                else:
                    self.train_status[metric_name].append(metric_value)
                if self.writer:
                    self.writer.add_scalar(metric_name+"/eval", metric_value, round + 1)
            if self.conf["train"]["save_best_model"]:   
                best_count = 0
                for metric_name, value_list in self.train_status.items():
                    if metrics[metric_name] == max(value_list):
                        best_count += 1
                if best_count == len(metrics):
                    print("Save the best model.")
                    torch.save(self.model.state_dict(), os.path.join("checkpoints", self.conf["task"]["name"]+".pth"))
      

class LefalPipeline(DeepTask):
    def __init__(self, dataset, conf, strategy):
        super().__init__(dataset, conf)
        self.client_nums = self.conf["federate"]["client_nums"]
        self.train_indices, self.test_indices = indices_train_test_split(list(range(len(self.dataset))))
        self.client_dict = self.get_client_dict(self.train_indices)
        self.clients = {}
        for client_idx, client_unlabeled_indices in self.client_dict.items():
            # The param 'dataset' is useless in clients' tasks
            # We need to manually input the unlabeled indices to the clients
            self.clients[client_idx] = getattr(activeUtils, strategy)(dataset, conf, client_unlabeled_indices)

    def get_client_dict(self, train_indices, seed=2020):
        client_indices_dict = partition(train_indices, self.client_nums)
        client_dict = {}
        for client, indices in client_indices_dict.items():
            # print(f"Client: {client}, Data numbers: {len(indices)}")
            sampler = SubsetRandomSampler(indices)
            client_dict[client] = indices
        return client_dict
    
    def start(self):
        print("Federated Round: ", self.conf["federate"]["round"])
        print("Client Numbers: ", self.client_nums)
        print("Model: ", self.conf["model"]["name"])
        print("Initial Budget: ", self.conf["active"]["init_budget"])
        print("Cycle Budget: ", self.conf["active"]["budget"])
        print("Cycle: ", self.conf["active"]["cycles"])
        print("Epochs Per Cycle: ", self.epochs)
         
    def run(self):
        # print("Train data numbers: ", len(train_indices))
        test_sampler = SubsetRandomSampler(self.test_indices)
        test_loader = DataLoader(self.dataset, batch_size=1, sampler=test_sampler)
        # self.model.train()
        for round in range(self.conf["federate"]["round"]):
            print(f"\n Federated Round {round + 1}")
            local_weights, local_loss = [], []
            for i in range(self.client_nums):
                # For each client, load the global weights first
                print(f"--------------------\nClient {i} local active training.")
                self.clients[i].model.load_state_dict(deepcopy(self.model.state_dict()))
                self.clients[i].model.to(self.clients[i].device)
                loss = self.clients[i].run_one_step()
                local_loss.append(deepcopy(loss))
                local_weights.append(deepcopy(self.clients[i].model.cpu().state_dict()))
            global_weight = average_weights(local_weights)
            self.model.load_state_dict(global_weight)
            self.model.to(self.device)
            global_loss = sum(local_loss) / len(local_loss)
            print("Train Loss: ", global_loss)
            if self.writer:
                self.writer.add_scalar("Loss/train", global_loss, round + 1)
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


class TefalPipeline(DeepTask):
    def __init__(self, dataset, conf, strategy):
        super().__init__(dataset, conf)
        self.client_nums = self.conf["federate"]["client_nums"]
        self.client_budget = self.conf["federate"]["client_budget"]
        self.train_indices, self.test_indices = indices_train_test_split(list(range(len(self.dataset))))
        self.client_dict = self.get_client_dict(self.train_indices)
        self.client_selection_status = [False for _ in range(self.client_nums)]
        self.discriminator = discriminator.Discriminator(z_dim=4)
        self.discriminator.to(self.device)
        self.dsc_optim = getattr(optimUtils, conf["discriminator"]["optimizer"]["name"](self.discriminator.parameters(), 
                                 conf["discriminator"]["optimizer"]))
        for client_idx, client_unlabeled_indices in self.client_dict:
            # The param 'dataset' is useless in clients' tasks
            # We need to manually input the unlabeled indices to the clients
            self.clients[client_idx] = TefalClient(dataset, conf) 

    def get_client_train_loader(self, train_indices, seed=2020):
        client_indices_dict = partition(train_indices, self.client_nums)
        client_dict = {}
        for client, indices in client_indices_dict.items():
            client_dict[client] = {}
            # print(f"Client: {client}, Data numbers: {len(indices)}")
            sampler = SubsetRandomSampler(indices)
            client_dict[client]["data_loader"] = DataLoader(self.dataset, batch_size=self.batch_size, sampler=sampler)
            client_dict[client]["data_nums"] = len(indices)
        return client_dict

    def init_client_selection(self, client_loss):
        loss_argsort = np.argsort(-np.array(client_loss))
        for i in loss_argsort[:self.client_budget]:
            self.client_selection_status[i] = True

    def dsc_pick_client(self, client_evals):
        raise NotImplementedError
        # self.discriminator.eval()
        # for client, evals in client_evals.items():

    def train_discriminator(self, client_evals):
        self.discriminator.train()
        selected_evals, not_selected_evals = [], []
        for client, evals in client_evals.items():
            if self.client_selection_status[client]:
                selected_evals.append(evals)
            else:
                not_selected_evals.append(evals)
        selected_evals, not_selected_evals = torch.tensor(selected_evals), torch.tensor(not_selected_evals)
        selected_evals, not_selected_evals = selected_evals.to(self.device), not_selected_evals.to(self.device)
        selected_pred, not_selected_pred = self.discriminator(selected_evals), self.discriminator(not_selected_evals)
        dsc_loss = nn.BCELoss()
        selected_gt = torch.ones(self.client_budget)
        not_selected_gt = torch.zeros(self.client_nums - self.client_budget)
        dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + \
                        self.bce_loss(unlabeled_preds, unlab_fake_preds)
        self.dsc_optim.zero_grad()
        dsc_loss.backward()
        self.dsc_optim.step()

    def run(self):
        train_indices, test_indices = indices_train_test_split(list(range(len(self.dataset))))
        # print("Train data numbers: ", len(train_indices))
        test_sampler = SubsetRandomSampler(test_indices)
        test_loader = DataLoader(self.dataset, batch_size=1, sampler=test_sampler)
        client_dict = self.get_client_train_loader(train_indices)
        # self.model.train()
        for _ in range(self.epochs):
            # print(f"\n Federated Round {self.cur_epoch}")
            local_weights, local_loss = [], []
            for i in range(self.client_nums):
                # For each client, load the global weights first
                # print(f"--------------------\nClient {i} local training.")
                self.clients[i].load_model_state_dict(deepcopy(self.model.state_dict()))
                self.clients[i].model.to(self.clients[i].device)
                loss = self.clients[i].train(client_dict[i]["data_loader"])
                local_loss.append(deepcopy(loss))
                local_weights.append(deepcopy(self.clients[i].model.cpu().state_dict()))
            global_weight = average_weights(local_weights)
            self.model.load_state_dict(global_weight)
            self.model.to(self.device)
            global_loss = sum(local_loss) / len(local_loss)
            # print("Train Loss: ", global_loss)
            if self.writer:
                self.writer.add_scalar("Loss/train", epoch_loss, self.cur_epoch)
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
        