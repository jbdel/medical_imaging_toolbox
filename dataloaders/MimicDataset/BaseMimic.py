from dataloaders.BaseDataset import BaseDataset
import numpy as np


class BaseMimic(BaseDataset):
    def __init__(self, task):
        super().__init__(task)

    @staticmethod
    def get_all_class_names_ordered():
        return ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
                'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
                'Pneumothorax', 'Support Devices']

    def get_tasks(self):
        return ['all', 'binary', 'six']

    def get_tree(self):
        if self.task == 'binary':
            return {'No Finding': {'No Finding'},
                    'Findings': {'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
                                 'Fracture', 'Lung Lesion', 'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
                                 'Pneumonia', 'Pneumothorax', 'Support Devices'}}

        elif self.task == 'six':
            # according to https://stanfordmlgroup.github.io/competitions/chexpert/img/figure1.png
            return {'No Finding': {'No Finding'},
                    'Support Devices': {'Support Devices'},
                    'Fracture': {'Fracture'},
                    'Lung Opacity': {'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Lung Lesion',
                                     'Atelectasis'},
                    'Enlarged Cardiomediastinum': {'Enlarged Cardiomediastinum', 'Cardiomegaly'},
                    'Pleural': {'Pleural Other', 'Pleural Effusion', 'Pneumothorax'}
                    }

        elif self.task == 'all':
            return {v: [v] for v in BaseMimic.get_all_class_names_ordered()}

    def get_encoded_label(self, label):
        # We can afford to put exceptions here before computing the label
        # For Mimic there is one when task is binary: both No findings and Support Devices can coexist
        # This would return the label [1,1] according to the binary tree.
        if self.task == 'binary' and (label['Support Devices'] == 1.0).all() and (label['No Finding'] == 1.0).all():
            label['Support Devices'] = 0.0
        return super().get_encoded_label(label)
