import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp
from sklearn.metrics import roc_auc_score, roc_curve, auc
import os

######
# Taken from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py #
#####

# Plot ROC curves for the multilabel problem
def plot_roc_multi(fpr, tpr, roc_auc, num_class, args, classes_name):
    # Outdir
    outdir = os.path.join(args.checkpoint_dir, 'plot_roc', 'epoch_' + str(args.current_epoch))
    os.makedirs(outdir, exist_ok=True)

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_class)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_class):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= num_class
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr

    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    lw = 2
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(num_class), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(classes_name[i], roc_auc['auc_' + str(i)]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(outdir, "plot_roc_multiclass"))


# Plot of a ROC curve for a specific class
def plot_roc(fpr, tpr, roc_auc, num_class, args, classes_name):
    # Outdir
    outdir = os.path.join(args.checkpoint_dir, 'plot_roc', 'epoch_' + str(args.current_epoch))
    os.makedirs(outdir, exist_ok=True)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'
                          ''.format(classes_name[num_class], roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(outdir, "plot_roc_" + str(num_class)))


# Get roc auc metrics
def get_roc_auc(y_true, y_pred, args, eval_loader):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    classes_name = eval_loader.dataset.get_classes_name()

    # Decision function
    if args.num_classes == 2:
        y_pred = F.softmax(torch.from_numpy(y_pred), dim=-1).numpy()
    else:
        y_pred = torch.sigmoid(torch.from_numpy(y_pred)).numpy()

    roc_auc["macro"] = roc_auc_score(y_true, y_pred, average='macro')
    roc_auc["micro"] = roc_auc_score(y_true, y_pred, average='micro')

    # Per class auroc
    for i in range(args.num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        auc_i = auc(fpr[i], tpr[i])
        roc_auc['auc_' + str(i)] = auc_i
        plot_roc(fpr[i], tpr[i], auc_i, i, args, classes_name)

    # Plotting all classes
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    plot_roc_multi(fpr, tpr, roc_auc, args.num_classes, args, classes_name)

    return roc_auc
