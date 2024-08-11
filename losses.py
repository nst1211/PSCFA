#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
torch.manual_seed(20230226)  # 设置CPU的随机种子
torch.cuda.manual_seed(20230226)  # 设置GPU的随机种子


#定义损失
class BinaryDiceLoss(nn.Module):
    """ Dice loss """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, input, target):
        assert input.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = nn.Sigmoid()(input)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1)#对应元素相乘再求和，统计1的数量，分子
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1)

        loss = 1 - (2 * num) / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))





if __name__ == "__main__":
    u = torch.Tensor([[[1.0, 0.0],
                       [0.0, 1.0],
                       [-1.0, 0.0],
                       [0.0, -1.0]],
                      [[1.0, 0.0],
                       [0.0, 1.0],
                       [-1.0, 0.0],
                       [0.0, -1.0]],
                      [[2.0, 0.0],
                       [0.0, 2.0],
                       [-2.0, 0.0],
                       [0.0, -4.0]]])

    v = torch.Tensor([[[0.0, 0.0],
                       [0.0, 2.0],
                       [-2.0, 0.0],
                       [0.0, -3.0]],
                      [[2.0, 0.0],
                       [0.0, 2.0],
                       [-2.0, 0.0],
                       [0.0, -4.0]],
                      [[1.0, 0.0],
                       [0.0, 1.0],
                       [-1.0, 0.0],
                       [0.0, -1.0]]])

    print("Input shape is (B,W,H):", u.shape, v.shape)
