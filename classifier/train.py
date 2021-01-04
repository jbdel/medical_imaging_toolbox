import torch
import torch.nn as nn
import numpy as np
import time
import os
from .utils import logwrite
from .metrics.metrics import get_metrics
import collections


def train(net, losses_fn, train_loader, eval_loader, optimizer, scheduler, args, data_args, model_args):
    logfile = open(
        args.output + "/" + args.name +
        '/log_run.txt',
        'w+'
    )

    logwrite(logfile, str(args), to_print=False)
    logwrite(logfile, "Total number of parameters : " + str(sum([p.numel() for p in net.parameters()]) / 1e6) + "M")

    args.no_improvement = 0
    args.best_metric_value = 0.0
    loss_dict = dict()
    summary = ''
    for epoch in range(0, args.epochs):
        args.current_epoch = epoch
        net.train()
        time_start = time.time()
        for step, sample in enumerate(train_loader):
            optimizer.zero_grad()
            pred = net(sample)

            losses = 0.0
            for key in losses_fn.keys():
                name, func, loss_args = losses_fn[key]
                loss = func(pred[key], sample[key].cuda(), **loss_args)
                losses += loss
                loss_dict[name] = loss.cpu().data.numpy()
            losses.backward()

            # Gradient norm clipping
            if args.grad_norm_clip > 0:
                nn.utils.clip_grad_norm_(
                    net.parameters(),
                    args.grad_norm_clip
                )

            optimizer.step()
            summary = "\r[Epoch {}][Step {}/{}] Loss: {}, Lr: {}, ES: {}/{} ({}: {:.2f}) - {:.2f} m remaining".format(
                args.current_epoch + 1,
                step,
                int(len(train_loader.dataset) / args.batch_size),
                ["{}: {:.2f}".format(k, v / args.batch_size) for k, v in loss_dict.items()],
                *[group['lr'] for group in optimizer.param_groups],
                args.no_improvement,
                args.early_stop,
                args.early_stop_metric,
                args.best_metric_value,
                ((time.time() - time_start) / (step + 1)) * (
                        (len(train_loader.dataset) / args.batch_size) - step) / 60,
            )
            print(summary, end='          ')

        time_end = time.time()
        elapse_time = time_end - time_start
        print('Finished in {}s'.format(int(elapse_time)))
        logwrite(logfile, summary)

        if epoch + 1 >= args.eval_start:
            metrics = evaluate(net, losses_fn, eval_loader, args)
            logwrite(logfile, metrics)
            metric_value = metrics[args.early_stop_metric]
            args.no_improvement += 1

            # Best model beaten
            if metric_value > args.best_metric_value:
                torch.save(
                    {
                        'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'args': args,
                        'data_args': data_args,
                        'model_args': model_args,
                        'metrics': metrics,
                    },
                    args.output + "/" + args.name +
                    '/best' + str(args.seed) + '.pkl'
                )
                args.no_improvement = 0
                args.best_metric_value = metric_value

            # Scheduler
            if args.use_scheduler:
                scheduler.step(metrics[args.early_stop_metric])

        # Early stop ?
        if args.early_stop == args.no_improvement:
            import sys
            os.rename(args.output + "/" + args.name +
                      '/best' + str(args.seed) + '.pkl',
                      args.output + "/" + args.name +
                      '/best' + str(args.seed) + '_' + str(args.best_metric_value) + '.pkl')
            print('Early stop reached')
            sys.exit()


def evaluate(net, losses_fn, eval_loader, args):
    print('Evaluation...')
    net.eval()
    with torch.no_grad():
        preds = collections.defaultdict(list)
        labels = collections.defaultdict(list)

        # Getting all prediction and labels
        for step, sample in enumerate(eval_loader):
            pred = net(sample)
            for key in losses_fn.keys():
                preds[key].append(pred[key].cpu().data.numpy())
                labels[key].append(sample[key].cpu().data.numpy())

        metrics = dict()
        for key in preds.keys():
            name, _, _ = losses_fn[key]
            pred = np.concatenate(preds[key])
            label = np.concatenate(labels[key])
            ret_metrics = get_metrics(losses_fn[key], pred, label, args, eval_loader)
            metrics = {**metrics, **ret_metrics}

    return metrics
