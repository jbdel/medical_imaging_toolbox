from .models.cnn import *
from .losses import *
from .metrics import *


def logwrite(log, s, to_print=True):
    if to_print:
        print(s)
    log.write(str(s) + "\n")


def get_losses_fn(cfg):
    return [eval(loss)(cfg) for loss in cfg.losses]

def get_metrics(cfg):
    return [eval(metric)(cfg, **cfg.metrics_params) for metric in cfg.metrics]


def get_model(cfg):
    if 'cosine' in cfg.losses and ('resnet' in cfg.model or 'densenet' in cfg.model):
        return CNNConstrained
    elif 'resnet' in cfg.model or 'densenet' in cfg.model:
        return CNN
    else:
        raise NotImplementedError
