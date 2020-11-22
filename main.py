from utils.dataset import ImageDataset, ImageSegDataset
import utils.data as data
from config import config
from torch.utils.data import DataLoader, SubsetRandomSampler
from model import resnet, lossnet, unet
import sampler
from torch import nn, optim
from utils.train import seg_train_epoch
import logging
from tqdm import tqdm
import torch
import random


# 配置训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO: 开发中
def deep_active_learn():
    # 获取文件和标签
    train_data_paths, train_labels = data.load_labeled_data_paths(config["data_root"]["train"])
    test_data_paths, test_labels = data.load_labeled_data_paths(config["data_root"]["test"])

    # 构建数据池，通过将labeled_indices传入sampler来模拟数据被oracle打标签
    data_pool = ImageDataset(train_data_paths, train_labels, data.image_resize)

    # 构造所有data的索引
    indices = list(range(len(data_pool)))
    random.shuffle(indices)

    # 初始化labeled数据
    k = config["active_learn"]["initial_labeled_nums"]
    labeled_indices = indices[:k]
    unlabeled_indices = indices[k:]

    # 通过labeled_indices构造Train DataLoader
    train_sampler = SubsetRandomSampler(labeled_indices)
    train_loader = DataLoader(data_pool, batch_size=config["deep_learn"]["batch_size"], sampler=train_sampler,
                              pin_memory=False) # 设备性能好时设置为True，加快数据转到GPU的速度

    # 定义任务模型
    task_model = resnet.ResNet18(num_classes=3)
    task_model.to(device)

    # 预训练
    task_model_dict = task_model.state_dict()
    pretrained_dict = torch.load(config["preprocess"]["pretrained_model_path"])
    parameter_dict = {k: v for k, v in pretrained_dict.items() if k in task_model_dict}
    task_model_dict.update(parameter_dict)
    task_model.load_state_dict(task_model_dict)

    # 损失预测模块
    loss_module = lossnet.LossNet()
    loss_module.to(device)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义优化器
    task_model_optimizer = optim.SGD(task_model.parameters(), lr=config["deep_learn"]["task_model"]["lr"],
                                     momentum=config["deep_learn"]["task_model"]["momentum"],
                                     weight_decay=config["deep_learn"]["task_model"]["weight_decay"])
    loss_module_optimizer = optim.SGD(task_model.parameters(), lr=config["deep_learn"]["loss_module"]["lr"],
                                     momentum=config["deep_learn"]["loss_module"]["momentum"],
                                     weight_decay=config["deep_learn"]["loss_module"]["weight_decay"])

    # for cycle in range(config["active_learn"]["cycles"]):
        
    # active_sampler = sampler.HybridSampler()


# TODO: 测试
# 分割深度学习调这个接口
def segmentation_pipeline(lr=1e-3, batch_size=8, epochs=50, test_size=0.2):
    # 获取训练数据和ground-truth掩膜
    data_paths, mask_paths = data.load_seg_data_paths(config["data_root"]["segmentation"])
    data_pool = ImageSegDataset(data_paths, mask_paths, data.image_resize, data.process_masks)

    # 用indices来划分训练和测试数据
    indices = list(range(len(data_pool)))
    random.shuffle(indices)
    train_indices, test_indices = indices[:int((1-test_size)*len(indices))], indices[int((1-test_size)*len(indices)):]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(data_pool, batch_size=config["segmentation"]["batch_size"], 
                              pin_memory=False, sampler=train_sampler)
    test_loader = DataLoader(data_pool, batch_size=config["segmentation"]["batch_size"],
                             pin_memory=False, sampler=test_sampler)

    net = unet.UNet(n_channels=3, n_classes=3)
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    # 训练
    # TODO: tqdm过程中展示acc和loss
    for epoch in range(epochs):
        epoch_loss = seg_train_epoch(net, train_loader, test_loader, device, criterion, optimizer, epoch)
        logging.info(f"Epoch: {epoch+1}, Training loss: {epoch_loss}")
        # 模型的weight可以通过net.state_dict()来拿


if __name__ == "__main__":
    # deep_active_learn()
    segmentation_pipeline()