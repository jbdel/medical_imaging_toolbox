import torch.nn as nn
import numpy as np
import torch


class BaseMetric(nn.Module):
    def __init__(self, cfg, key='key'):
        super(BaseMetric, self).__init__()
        self.key = key
        self.cfg = cfg

    def forward(self, input, target):
        assert isinstance(input, dict), 'input is not a dictionary'
        assert isinstance(target, dict), 'input is not a dictionary'

        input, target = input[self.key], target[self.key]

        if isinstance(input, (list)):
            input = np.array(input)
        if isinstance(target, (list)):
            target = np.array(target)

        return input, target
