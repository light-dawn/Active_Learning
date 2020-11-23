import torch
from tqdm import tqdm


# TODO: 训练过程中做validation
def train_epoch(net, train_loader, device, criterion, optimizer, cur_epoch):
    net.train()
    epoch_loss = 0
    process = tqdm(train_loader, leave=True)

    for images, targets in process:
        # print(images.size())
        # print(masks.size())
        # 读取loader中的数据
        images = images.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.long)
        assert images.shape[1] == net.n_channels, f"图像通道数与网络通道数不匹配"

        # 预测并计算loss
        prediction = net(images)
        loss = criterion(prediction, targets)
        epoch_loss += loss.item()

        # 消除上一次梯度，然后反向传播并更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        process.set_description(f"Train epoch: {cur_epoch + 1}, Loss: {loss.item()}")
    
    return epoch_loss


def validation(net, test_loader, device):
    net.eval()
    total, correct = 0, 0
    process = tqdm(test_loader, leave=True)
    with torch.no_grad():
        for images, targets in process:
            images = images.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.long)
            assert images.shape[1] == net.n_channels, f"图像通道数与网络通道数不匹配"

            prediction = net(images)




