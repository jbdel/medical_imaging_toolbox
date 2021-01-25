from __future__ import print_function
import os
import pickle
import sys
import torch
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from tqdm import tqdm
from .BaseMimic import BaseMimic

data_root = './data/mimic-crx/'
split_file = 'mimic-cxr-2.0.0-split.csv'
label_file = 'mimic-cxr-2.0.0-chexpert.csv'
report_file = 'mimic-crx-reports/mimic_cxr_sectioned.csv'
image_folder = 'mimic-crx-jpg-images/'


def get_transforms(name):
    if name == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


class OldMimicDataset(BaseMimic):
    def __init__(self, split, return_image=False, return_label=False, return_report=False, task='all', **kwargs):
        super(OldMimicDataset, self).__init__(task)
        assert split in ['train', 'val', 'test']
        self.split = split
        # bleh
        if self.split == 'val':
            self.split = 'validate'
            split = 'validate'
        self.return_image = return_image
        self.return_report = return_report
        self.return_label = return_label
        self.task = task

        # Open split file
        df_samples = pd.read_csv(os.path.join(data_root, split_file))
        df_samples = df_samples.loc[df_samples['split'] == split]
        # Open label file
        df_labels = pd.read_csv(os.path.join(data_root, label_file))
        # Open report file
        df_reports = pd.read_csv(os.path.join(data_root, report_file))

        # Building set
        self.samples = {}
        self.set_file = os.path.join(data_root, split + '_set.pkl')
        if os.path.exists(self.set_file):
            print("Loading " + split + " dataset")
            self.samples = pickle.load(open(self.set_file, 'rb'))
        else:
            print("Building " + split + " dataset")
            excluded = 0
            for index, row in tqdm(df_samples.iterrows(), total=df_samples.shape[0]):
                # Keys are a  triple (subject_id, study_id, dicom_id)
                key = (row['subject_id'], row['study_id'], row['dicom_id'])

                # Fetch label
                label = df_labels.loc[(df_labels['subject_id'] == row['subject_id']) &
                                      (df_labels['study_id'] == row['study_id'])].fillna(0)

                # Fetch report
                report = df_reports.loc[df_reports['study'] == ('s' + str(row['study_id']))]
                txt = ''
                for section in ['impression', 'findings', 'last_paragraph', 'comparison']:
                    if not report[section].isna().any():
                        txt = ''.join(report[section].values)
                        break

                if not (True in label.values) or report.empty or txt == '':
                    excluded += 1
                    # open('excluded.txt', 'a+').write(str(key)+"\n")
                    continue
                self.samples[key] = (label, txt)

            print("Excluded " + str(excluded) + " samples (no label or report). Current samples set is ",
                  len(self.samples))
            # save constructed set
            pickle.dump(self.samples, open(self.set_file, 'wb'))

        self.keys = list(self.samples.keys())
        self.transform = get_transforms(split)

    def __getitem__(self, idx):
        subject_id, study_id, dicom_id = self.keys[idx]
        sample = self.samples[self.keys[idx]]  # A sample is a [label, report]
        img = torch.tensor(0)
        label = torch.tensor(0)
        report = torch.tensor(0)
        import time
        if self.return_image:
            img_path = os.path.join(data_root,
                                    image_folder,
                                    'p' + str(subject_id)[:2],  # 10000032 -> p10
                                    'p' + str(subject_id),
                                    's' + str(study_id),
                                    str(dicom_id) + '.jpeg'
                                    )
            try:
                img = self.transform(Image.open(img_path).convert('RGB'))
            except FileNotFoundError:
                print('image not found for key', self.keys[idx], img_path)
                raise

        if self.return_report:
            _, report = sample

        if self.return_label:
            label, _ = sample
            if self.task == 'binary' and (label['Support Devices'] == 1.0).all() and (label['No Finding'] == 1.0).all():
                label['Support Devices'] = 0.0
            vector = np.zeros(len(self.task_classes))
            for (c, l) in label.iteritems():
                if (l > 0.0).all():
                    try:
                        pos = self.pos_label_task[self.super_label[c]]
                        vector[pos] = 1.
                    except KeyError:
                        continue
            label = vector.astype(np.float)

        return {'idx': idx,
                'key': self.keys[idx],
                'report': report,
                'img': img,
                'label': label}

    def __len__(self):
        return len(self.keys)


if __name__ == '__main__':
    d = OldMimicDataset("train", return_image=True,
                     return_label=True,
                     return_report=True,
                     task='six')
    for s in tqdm(d):
        continue
