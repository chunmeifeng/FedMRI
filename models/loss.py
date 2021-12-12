import torch
from torch import nn
import torch.nn.functional as F

class LossWrapper(nn.Module):
    def __init__(self, args):
        super(LossWrapper, self).__init__()
        self.l1_loss = nn.MSELoss()

    def forward(self, outputs, targets):
        l1_loss = self.l1_loss(outputs, targets)
        loss = l1_loss
        return {'l1_loss' : l1_loss, 'loss': loss}


def Criterion(args):
    return LossWrapper(args)