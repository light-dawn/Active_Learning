import torch
from torch.autograd import Function


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


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)