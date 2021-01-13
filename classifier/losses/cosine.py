import torch.nn as nn
import torch


class CosineLoss(nn.Module):
    def __init__(self, cfg):
        super(CosineLoss, self).__init__()
        self.func = nn.CosineEmbeddingLoss(reduction="sum")
        self.key = 'cosine'

    def forward(self, input, target):
        input, target = input[self.key], target[self.key]
        target = target.to(device=input.device)
        return self.func(input, target, target=torch.ones(input.size()[0]).cuda())
