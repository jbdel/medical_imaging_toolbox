import torch.nn as nn
import torch

def get_losses_fn(args):
    loss_fn = dict()
    if 'classification' in args.losses:
        loss_fn['label'] = ('classification', nn.BCEWithLogitsLoss(reduction="sum"), {})
    if 'cosine' in args.losses:
        loss_fn['vector'] = ('cosine', nn.CosineEmbeddingLoss(reduction="sum"), {'target': torch.ones(args.batch_size).cuda()})

    return loss_fn