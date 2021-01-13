import argparse, os, random
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from dataloaders import *
from torch.optim.lr_scheduler import *
import torch.optim as optim
from torch.utils.data import DataLoader

from .train import train
from .utils import get_losses_fn, get_model, get_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    torch.backends.cudnn.benchmark = True

    # DataLoader
    train_dset: BaseDataset = eval(cfg.dataset)('train', **cfg.dataset_params)
    eval_dset: BaseDataset = eval(cfg.dataset)('val', **cfg.dataset_params)
    test_dset: BaseDataset = eval(cfg.dataset)('test', **cfg.dataset_params)
    cfg.dataset_params.num_classes = train_dset.num_classes
    cfg.dataset_params.task_classes = train_dset.task_classes

    n_gpus = torch.cuda.device_count()
    train_loader = DataLoader(train_dset,
                              cfg.hyperparameter.batch_size,
                              shuffle=True,
                              num_workers=n_gpus * 4,
                              pin_memory=True,
                              drop_last=True)

    eval_loader = DataLoader(eval_dset,
                             int(cfg.hyperparameter.batch_size / 2),
                             num_workers=n_gpus * 4,
                             pin_memory=True)

    # test_loader = DataLoader(test_dset,
    #                          int(cfg.hyperparameter.batch_size / 2),
    #                          num_workers=n_gpus * 4,
    #                          pin_memory=True)

    dataset_keys = train_dset.__getitem__(0).keys()
    print('Using dataset', type(train_dset).__name__, 'with task', train_dset.task, 'returning batch with keys',
          dataset_keys)
    # Net
    net_func = get_model(cfg)
    cfg.model_params = OmegaConf.merge({
        'name': cfg.model,
        'num_classes': train_dset.num_classes,
        'net_func': net_func.__name__
    }, cfg.model_params)

    net = net_func(**cfg.model_params).cuda()
    print('Using network', type(net).__name__, 'with input', net.forward_input_keys, 'returning',
          net.get_forward_output_keys(), 'with', net.num_classes,
          'output neurons')

    # Losses
    losses_fn = get_losses_fn(cfg)
    print('Using losses', [type(loss).__name__ for loss in losses_fn], "needing the model to return",
          [loss.key for loss in losses_fn])

    # Metrics
    metrics_fn = get_metrics(cfg)
    print('Using metrics', [type(metric).__name__ for metric in metrics_fn], "needing the model to return",
          [metric.key for metric in metrics_fn])

    # Checking configs OK
    assert set(net.get_forward_output_keys()) == set([loss.key for loss in losses_fn]), 'Losses and model dont match'
    assert all([key in dataset_keys for key in net.forward_input_keys]), 'Model and dataset dont match'
    print('Everything matches')

    # Create Checkpoint dir
    cfg.checkpoint_dir = os.path.join(cfg.experiment.output_dir, cfg.experiment.name)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    optimizer = optim.AdamW(net.parameters(), lr=cfg.hyperparameter.lr_base)
    print('Using optimizer', optimizer)

    scheduler = None
    if cfg.scheduler.use_scheduler:
        scheduler = eval(cfg.scheduler.name)(optimizer, **cfg.scheduler_params)
        print('Using scheduler', scheduler)

    net = nn.DataParallel(net)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # Run training
    eval_accuracies = train(net,
                            losses_fn,
                            metrics_fn,
                            train_loader,
                            eval_loader,
                            optimizer,
                            scheduler,
                            cfg,
                            )
