import torch.nn as nn
from torchvision.models import *
from .basemodel import BaseModel


class CNN(BaseModel):
    def __init__(self, backbone, num_classes, pretrained=True, **kwargs):
        super(CNN, self).__init__(num_classes)
        self.net = eval(backbone)(pretrained=pretrained)

        # Overriding classifier
        if 'densenet' in backbone:
            self.fc_name = 'classifier'
        elif 'resnet' in backbone:
            self.fc_name = 'fc'
        else:
            raise NotImplementedError

        self.in_features = getattr(self.net, self.fc_name).in_features
        setattr(self.net, self.fc_name, nn.Linear(self.in_features, self.num_classes))

    def forward(self, sample):
        return {'label': self.net(sample['img'].cuda())}

    def get_forward_keys(self):
        return ['label']


class CNNConstrained(CNN):
    def __init__(self, backbone, num_classes, pretrained=True, vector_size=300, **kwargs):
        super(CNNConstrained, self).__init__(backbone, num_classes, pretrained, **kwargs)
        setattr(self.net, self.fc_name, nn.Linear(self.in_features, vector_size))
        self.out = nn.Linear(vector_size, self.num_classes)

    def forward(self, sample):
        vector = self.net(sample['img'].cuda())
        label = self.out(vector)
        return {'vector': vector,
                'label': label}

    def get_forward_keys(self):
        return ['label', 'vector']
