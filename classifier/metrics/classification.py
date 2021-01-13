import torch.nn.functional as F
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
from .plot import plot_roc, plot_roc_multi
from .basemetric import BaseMetric
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


class ClassificationMetric(BaseMetric):
    def __init__(self, cfg, decision_function='sigmoid'):
        super(ClassificationMetric, self).__init__(cfg, key='label')
        self.decision_function = decision_function

        # Getting logit function and prediction function from decision param
        if self.decision_function == 'softmax':
            self.pred_fn = lambda x: np.argmax(x, axis=-1)
            self.logit_fn = lambda x: F.softmax(torch.from_numpy(x), dim=-1).numpy()
        elif self.decision_function == 'sigmoid':
            self.pred_fn = lambda x: (x > 0)
            self.logit_fn = lambda x: torch.sigmoid(torch.from_numpy(x)).numpy()
        else:
            raise NotImplementedError(self.decision_function)

    def forward(self, input, target):
        input, target = super().forward(input, target)
        # AUC
        roc_auc = self.get_roc_auc(target, input)
        accuracy_f1 = self.accuracy_f1(target, input)
        return {**accuracy_f1, **roc_auc}

    def accuracy_f1(self, y_true, y_pred):
        metrics = dict()
        y_pred = self.pred_fn(y_pred)
        y_true = self.pred_fn(y_true)

        metrics['accuracy'] = np.mean(y_pred == y_true)
        metrics['f1_score'] = f1_score(y_true, y_pred, average=None)
        metrics['f1_score_weighted'] = f1_score(y_true, y_pred, average='weighted')
        metrics['f1_score_macro'] = f1_score(y_true, y_pred, average='macro')
        metrics['classification_report'] = classification_report(y_true, y_pred,
                                                                 target_names=self.cfg.dataset_params.task_classes)
        return metrics

    def get_roc_auc(self, y_true, y_pred):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        task_classes = self.cfg.dataset_params.task_classes
        num_classes = self.cfg.dataset_params.num_classes

        # Getting logits from decision function
        y_pred = self.logit_fn(y_pred)

        roc_auc["macro"] = roc_auc_score(y_true, y_pred, average='macro')
        roc_auc["micro"] = roc_auc_score(y_true, y_pred, average='micro')

        # Per class auroc
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
            auc_i = auc(fpr[i], tpr[i])
            roc_auc['auc_' + str(i)] = auc_i
            plot_roc(fpr[i], tpr[i], auc_i, i, self.cfg, task_classes)

        # Plotting all classes
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
        plot_roc_multi(fpr, tpr, roc_auc, num_classes, self.cfg, task_classes)

        return roc_auc
