import numpy as np
import torch
from sklearn.metrics import f1_score
from .roc_auc import get_roc_auc


def amax(x):
    return np.argmax(x, axis=-1)


def multi_label(x):
    return (x > 0)


def get_metrics(losses_fn, pred, label, args, eval_loader):
    """Compute metrics

        Parameters
        ----------
        losses_fn : triplet (name, loss_func, loss_args). Loss used during training to get this output
        pred : numpy array
        label : numpy array

    """
    metrics = dict()
    name, loss_func, loss_args = losses_fn
    if name == 'classification':
        # Val loss
        metrics['label_loss'] = loss_func(torch.from_numpy(pred),
                                          torch.from_numpy(label),
                                          **loss_args).cpu().item()
        # Auc
        roc_auc = get_roc_auc(label, pred, args, eval_loader)
        metrics = {**metrics, **roc_auc}
        # Accuracy and F1
        pred = eval(args.pred_func)(pred)
        label = eval(args.pred_func)(label)
        metrics['accuracy'] = np.mean(pred == label)
        metrics['f1_score'] = f1_score(label, pred, average=None)
        metrics['f1_score_weighted'] = f1_score(label, pred, average='weighted')
        metrics['f1_score_macro'] = f1_score(label, pred, average='macro')

    if name == 'vector':
        metrics['cosine_loss'] = loss_func(torch.from_numpy(pred).cuda(),
                                           torch.from_numpy(label).cuda(),
                                           target=torch.ones(pred.shape[0]).cuda()).cpu().item()

    return metrics
