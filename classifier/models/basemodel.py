import torch.nn as nn
import abc


class BaseModel(nn.Module):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def forward(self, sample) -> dict:
        """forward method, must implement"""
        raise NotImplementedError('users must define forward to use this base class')

    @abc.abstractmethod
    def get_forward_keys(self) -> list:
        """return the keys of the dict returned by forward, must implement"""
        raise NotImplementedError('users must define get_forward_keys to use this base class')
