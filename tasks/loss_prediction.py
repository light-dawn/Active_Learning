from tasks.base_tasks import DeepActiveTask
from sampler import LossPredictionSampler

from models import unet, lossnet
from loss.loss_prediction_loss import loss_prediction_loss
import json

from utils.data import DataUtils
from utils.dataset import ImageSegDataset
import random
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import nn, optim
import os


class LossPredSegPipeline(DeepActiveTask):
    def __init__(self, task_name, model, dataset, optimizer, criterion, epochs, batch_size, 
                 init_budget, budget, cycles):
        super().__init__(task_name, model, dataset, optimizer, criterion, epochs, batch_size, 
                         init_budget, budget, cycles, lossnet_feature_sizes, lossnet_num_channels)
        self.sampler = LossPredictionSampler(self.budget)
        self.lossnet = lossnet.LossNet(feature_sizes=lossnet_feature_sizes, num_channels=lossnet_num_channels)
        self.lossnet.to(self.device)
    
    # Jointly train the lossnet along with the task model
    def train_epoch(self, train_loader):
        self.model.train()
        self.lossnet.train()
        epoch_loss = 0
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
            epoch_loss += task_loss.item()

            pred_loss = self.lossnet(feature_maps)
            pred_loss.view(pred_loss.size(0))

            target_loss = torch.sum(task_loss) / task_loss.size(0)
            loss_pred_loss = loss_prediction_loss(pred_loss, target_loss)

            self.optimizer.zero_grad()
            task_loss.backward()
            loss_pred_loss.backward()
            self.optimizer.step()

            process.set_description(f"Train epoch: {self.cur_epoch}, Loss: {loss.item():.5f}")
        return epoch_loss


    def run(self):
        for cycle in range(self.cycles):
            print("Cycle: ", cycle + 1)
            # 构造DataLoader
            labeled_loader = self.get_cur_data_loader(part="labeled")
            unlabeled_loader = self.get_cur_data_loader(part="unlabeled")
            self.train(labeled_loader)
            query_indices = self.sampler.sample(unlabeled_loader, self.lossnet)
            self.query_and_move(query_indices)
            print("Query and move data...")


def loss_pred_sampling():
    # 之后调整为读取控制台参数
    with open("loss_pred_cfg.json", "r") as f:
        config = json.loads(f.read())
    # print(config)
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
    task = LossPredSegPipeline("active_seg_demo", model, dataset, optimizer, criterion, epochs, batch_size, 
                             init_budget, budget, cycles, lossnet_feature_sizes, lossnet_num_channels)
    task.run()


if __name__ == "__main__":
    random_sampling()




