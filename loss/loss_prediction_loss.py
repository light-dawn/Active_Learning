import torch


def loss_prediction_loss(inputs, target, margin=1.0, reduction='mean'):
    assert len(inputs) % 2 == 0, 'the batch size is not even.'
    assert inputs.shape == inputs.flip(0).shape
    # flip()翻转
    inputs = (inputs - inputs.flip(0))[
            :len(inputs) // 2]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()

    # 将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量。
    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1  # 1 operation which is defined by the authors
    loss = None

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * inputs, min=0))
        loss = loss / inputs.size(0)  # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * inputs, min=0)
    else:
        NotImplementedError()
    return loss