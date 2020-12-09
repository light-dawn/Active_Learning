from tasks.base_tasks import DeepTask
from models import unet
import json
from utils.data import DataUtils
from utils.dataset import ImageSegDataset
from utils.federate import partition, average_weights
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import nn, optim
import os
import random
from copy import deepcopy


class FedSegPipeline(DeepTask):
    def __init__(self, task_name, model, dataset, optimizer, criterion, epochs, batch_size, client_nums, fed_round, 
                 save_weight=False, write_tensorboard=False):
        super().__init__(task_name, model, dataset, optimizer, criterion, fed_round, batch_size, write_tensorboard)
        print(self.epochs)
        self.client_nums = client_nums
        self.save_weight = save_weight
        self.clients = {}
        for i in range(self.client_nums):
            # the param 'dataset' is useless in clients' tasks
            # self.clients[i] = DeepTask(task_name+"client"+str(i), deepcopy(model), dataset, optimizer, criterion, epochs, batch_size, 
            #                            write_tensorboard=False)
            self.clients[i] = DeepTask("", deepcopy(model), dataset, deepcopy(optimizer), deepcopy(criterion), 
                                       epochs, batch_size, write_tensorboard=False)

    def get_train_test_indices(self, test_size=0.2, seed=2020):
        random.seed(seed)
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        # print(indices)
        train_indices, test_indices = indices[:int((1-test_size)*len(indices))], indices[int((1-test_size)*len(indices)):]
        return train_indices, test_indices

    def get_client_train_loader(self, train_indices, seed=2020):
        client_indices_dict = partition(train_indices, self.client_nums)
        client_trainloader_dict = {}
        for client, indices in client_indices_dict.items():
            # print(f"Client: {client}, Data numbers: {len(indices)}")
            sampler = SubsetRandomSampler(indices)
            client_trainloader_dict[client] = DataLoader(self.dataset, batch_size=self.batch_size, sampler=sampler)
        return client_trainloader_dict
        
    def run(self):
        train_indices, test_indices = self.get_train_test_indices()
        print("Train data numbers: ", len(train_indices))
        test_sampler = SubsetRandomSampler(test_indices)
        test_loader = DataLoader(self.dataset, batch_size=1, sampler=test_sampler)
        client_train_loaders = self.get_client_train_loader(train_indices)
        self.model.train()
        for _ in range(self.epochs):
            print(f"\n Federated Round {self.cur_epoch}")
            local_weights, local_loss = [], []
            for i in range(self.client_nums):
                # For each client, load the global weights first
                print(f"--------------------\nClient {i} local training.")
                self.clients[i].load_model_state_dict(deepcopy(self.model.state_dict()))
                self.clients[i].model.to(self.clients[i].device)
                loss = self.clients[i].train(client_train_loaders[i])
                local_loss.append(deepcopy(loss))
                local_weights.append(deepcopy(self.clients[i].model.cpu().state_dict()))
                self.clients[i].cur_epoch += 1
            global_weight = average_weights(local_weights)
            self.model.load_state_dict(global_weight)
            self.model.to(self.device)
            global_loss = sum(local_loss) / len(local_loss)
            print("Train Loss: ", global_loss)
            if self.writer:
                self.writer.add_scalar("Loss/train", epoch_loss, self.cur_epoch)
            metrics = self.eval_seg(test_loader)
            for metric_name, metric_value in metrics.items():
                print(f"{metric_name}: {metric_value}")
                if self.writer:
                    self.writer.add_scalar(metric_name+"/eval", metric_value, self.cur_epoch)
            if self.save_weight:
                self.save_model_state_dict(os.path.join("checkpoints", self.task_name+"_Epoch_"+str(self.cur_epoch)+".pth"))
            self.cur_epoch += 1
        
def pipeline(config):
    # Model
    model = unet.UNet(n_channels=config["model"]["n_channels"], n_classes=config["model"]["n_classes"])
    # Dataset
    data_paths, mask_paths = DataUtils.load_seg_data_paths(config["data"]["data_root"], sep="\\")
    print("Data path numbers: ", len(data_paths))
    dataset = ImageSegDataset(data_paths, mask_paths, DataUtils.image_resize, DataUtils.process_masks)
    print("Dataset size: ", len(dataset))
    # Optimizer
    optimizer = optim.RMSprop(model.parameters(), lr=config["optimizer"]["lr"], 
                              weight_decay=config["optimizer"]["weight_decay"], 
                              momentum=config["optimizer"]["momentum"])
    # Loss Function
    if model.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    epochs, batch_size = config["train"]["epochs"], config["train"]["batch_size"]
    # Federated Setting
    client_nums, fed_round = config["federate"]["client_num"], config["federate"]["round"]
    # Task
    task = FedSegPipeline("fed_seg_trail", model, dataset, optimizer, criterion, epochs, batch_size, client_nums, fed_round)
    task.run()
    task.end()


if __name__ == "__main__":
    pipeline()