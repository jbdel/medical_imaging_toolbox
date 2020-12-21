import torch.nn as nn
from torchvision.models import *
from .basemodel import BaseModel


class CNN(BaseModel):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.net = eval(args.model)(pretrained=True)

        # Overriding classifier
        if 'densenet' in args.model:
            self.fc_name = 'classifier'
        elif 'resnet' in args.model:
            self.fc_name = 'fc'
        else:
            raise NotImplementedError

        self.in_features = getattr(self.net, self.fc_name).in_features
        setattr(self.net, self.fc_name, nn.Linear(self.in_features, args.num_classes))

    def forward(self, sample):
        return {'label': self.net(sample['img'].cuda())}

    def get_forward_keys(self):
        return ['label']


class CNNConstrained(CNN):
    def __init__(self, args):
        super(CNNConstrained, self).__init__(args)
        setattr(self.net, self.fc_name, nn.Linear(self.in_features, args.vector_size))
        setattr(self.net, 'out', nn.Linear(args.vector_size, args.num_classes))

    def forward(self, sample):
        vector = self.net(sample['img'].cuda())
        label = self.out(vector)
        return {'vector': vector,
                'label': label}

    def get_forward_keys(self):
        return ['label', 'vector']
