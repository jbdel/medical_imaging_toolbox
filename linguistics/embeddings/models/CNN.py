import torch.nn as nn
import os
import torch


class CNN(nn.Module):
    def __init__(self, cfg):
        super(CNN, self).__init__()
        self.cfg = cfg
        checkpoint = cfg.model.checkpoint
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(checkpoint)

        ckpt = torch.load(checkpoint)
        network_name = ckpt['model_name'].__name__
        self.net = eval(network_name)(**ckpt['model_args'])
        self.net = nn.DataParallel(self.net)
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.cuda()
        self.net.eval()

        if 'densenet' in network_name.lower():
            fc_name = 'classifier'
        elif 'resnet' in network_name.lower():
            fc_name = 'fc'
        else:
            raise NotImplementedError
        self.fc = getattr(self.net.module.net, fc_name)

    def forward(self, sample):
        with torch.no_grad():
            vector = torch.zeros(self.fc.in_features)

            def hook(m, i, o): vector.copy_(i[0].squeeze().data)

            self.fc.register_forward_hook(hook)

            sample['img'] = sample['img'].unsqueeze(0)
            _ = self.net(sample)
            return vector.cpu().data.numpy()
