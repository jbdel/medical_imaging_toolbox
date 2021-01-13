import torch.nn as nn
import numpy as np
import torch


class BaseLoss(nn.Module):
    def __init__(self, cfg, key='key'):
        super(BaseLoss, self).__init__()
        self.key = key
        self.cfg = cfg

    def forward(self, input, target):
        assert isinstance(input, dict), 'input is not a dictionary'
        assert isinstance(target, dict), 'input is not a dictionary'

        input, target = input[self.key], target[self.key]
        if isinstance(input, (list, np.ndarray)):
            input = torch.as_tensor(np.array(input))
        if isinstance(target, (list, np.ndarray)):
            target = torch.as_tensor(np.array(target))
        target = target.to(device=input.device)
        return input, target
