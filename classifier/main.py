import argparse, os, random
import numpy as np
import torch
import torch.nn as nn

from dataloaders.MimicDataset import MimicDataset
import torch.optim as optim
from torch.utils.data import DataLoader

from .train import train
from .losses import get_losses_fn
from .models.utils import get_model


def parse_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model', type=str, default="resnet18")
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--dataset', type=str, default="MimicDataset")
    parser.add_argument('--losses', action='store', type=str, nargs='+')

    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--vector_size', type=int, default=300)
    parser.add_argument('--pred_func', type=str, default="amax")

    # Data
    parser.add_argument('--vector_folder', type=str, default='mimic-crx-vectors/')
    parser.add_argument('--return_image', type=bool, default=False)
    parser.add_argument('--return_report', type=bool, default=False)
    parser.add_argument('--return_label', type=bool, default=False)

    # Training
    parser.add_argument('--use_scheduler', type=bool, default=False)
    parser.add_argument('--lr_base', type=float, default=0.0001)
    parser.add_argument('--lr_max', type=float, default=0.02)
    parser.add_argument('--grad_norm_clip', type=float, default=-1)

    parser.add_argument('--epochs', type=int, default=9999)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--eval_start', type=int, default=0)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--early_stop_metric', type=str, default='auc_macro')

    parser.add_argument('--seed', type=int, default=random.randint(0, 9999))
    parser.add_argument('--output', type=str, default='checkpoints/')
    parser.add_argument('--name', type=str, default='exp0/')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Base on args given, compute new args
    args = parse_args()

    # Seed
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(args.seed)
    else:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # DataLoader
    data_args = {'return_image': args.return_image,
                 'return_label': args.return_label,
                 'return_report': args.return_report,
                 'task_binary': args.num_classes == 2}
    
    train_dset = eval(args.dataset)('train', **data_args)
    eval_dset = eval(args.dataset)('validate', **data_args)
    test_dset = eval(args.dataset)('test', **data_args)

    train_loader = DataLoader(train_dset,
                              args.batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True,
                              drop_last=True)

    eval_loader = DataLoader(eval_dset,
                             args.batch_size,
                             num_workers=4,
                             pin_memory=True)

    test_loader = DataLoader(test_dset,
                             args.batch_size,
                             num_workers=4,
                             pin_memory=True)

    # Net
    net = get_model(args)
    print('Using network ', type(net).__name__, 'returning ', net.get_forward_keys())
    # Losses
    losses_fn = get_losses_fn(args)
    print('Using losses', [v[0] for _, v in losses_fn.items()], "needing the model to return", list(losses_fn.keys()))

    # Checking config OK
    assert set(net.get_forward_keys()) == set((losses_fn.keys())), 'Losses and model returns do not match'

    # Create Checkpoint dir
    args.checkpoint_dir = os.path.join(args.output, args.name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    optimizer = optim.Adam(net.parameters(), lr=args.lr_base)
    scheduler = None
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=0.1,
                                                               patience=5,
                                                               threshold=0.0001)
        print('Using scheduler', scheduler)

        # scheduler = optim.lr_scheduler.CyclicLR(optimizer,
        #                                         base_lr=args.lr_base,
        #                                         max_lr=args.lr_max,
        #                                         step_size_up=(len(train_loader.dataset) / args.batch_size) * 2,
        #                                         mode='triangular2',
        #                                         cycle_momentum=False)

    net = nn.DataParallel(net)
    net.cuda()
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.checkpoint is not None:
        print("Let's load checkpoint", str(args.checkpoint))
        ckpt = torch.load(args.checkpoint)
        net.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        args = ckpt['args']

    # Run training
    eval_accuracies = train(net, losses_fn, train_loader, eval_loader, args, optimizer, scheduler)