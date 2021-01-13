import torch.nn as nn
import numpy
import torch
from .baseloss import BaseLoss


class ClassificationLoss(BaseLoss):
    def __init__(self, cfg):
        super(ClassificationLoss, self).__init__(cfg, key='label')
        self.func = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(self, input, target):
        input, target = super().forward(input, target)
        return self.func(input, target)
