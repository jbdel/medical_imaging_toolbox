import torch.nn as nn
import abc
import numpy as np
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    __metaclass__ = abc.ABC

    def __init__(self, task='all'):
        super().__init__()
        assert task in self.get_tasks(), 'task {} unknown'.format(task)
        self.task = task

        self.task_classes = list(self.get_classes())
        self.num_classes = len(self.task_classes)

        self.pos_label = {c: i for i, c in enumerate(self.get_classes())}
        self.super_label = {c: self.get_super_label(c) for c in self.get_all_class_names_ordered()}

    @staticmethod
    @abc.abstractmethod
    def get_all_class_names_ordered():
        """Returns all class names ordered according to label distribution"""
        raise NotImplementedError('users must define get_all_class_names_ordered to use this base class')

    def get_tasks(self):
        """Returns the available task variants for the dataset. Default: use 'all' labels"""
        return ['all']

    def get_classes(self):
        """Returns available classes according to the task variant 'task'"""
        return self.get_tree().keys()

    def get_tree(self):
        """Returns a dictionary key (string): value (list) containing the class-tree"""
        if self.task == 'all':
            return {v: [v] for v in self.get_all_class_names_ordered()}

    def get_super_label(self, c):
        """Returns the superclass of class 'c' given task 'task'"""
        for k, v in self.get_tree().items():
            if c in v:
                return k

    def get_encoded_label(self, label):
        """Given a pandas Series (a row), return one hot or multi-hot vector. Complexity O(len(label))"""
        vector = np.zeros(len(self.task_classes))
        for (c, l) in label.iteritems():
            if (l > 0.0).all():
                try:
                    pos = self.pos_label[self.super_label[c]]
                    vector[pos] = 1.
                except KeyError:
                    continue
        return vector
