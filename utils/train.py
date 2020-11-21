# TODO: 训练过程中做validation
def seg_train_epoch(net, train_loader, device, criterion, optimizer):
    epoch_loss = 0
    for images, masks in train_loader:
        # 读取loader中的数据
        images = images.to(device)
        true_masks = masks.to(device)
        assert images.shape[1] == net.n_channels, f"图像通道数与网络通道数不匹配"

        # 预测并计算loss
        predicted_masks = net(images)
        loss = criterion(predicted_masks, true_masks)
        epoch_loss += loss.item()

        # 消除上一次梯度，然后反向传播并更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return epoch_loss


