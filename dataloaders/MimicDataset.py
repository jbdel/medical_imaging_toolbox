from __future__ import print_function
import os
import pickle
import numpy as np
import sys
from PIL import Image
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

CLASS_NAMES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", "Fracture",
               "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
               "Pneumothorax", "Support Devices"]

data_root = './data/'
split_file = 'mimic-cxr-2.0.0-split.csv'
label_file = 'mimic-cxr-2.0.0-chexpert.csv'
report_file = 'mimic-crx-reports/mimic_cxr_sectioned.csv'
image_folder = 'mimic-crx-images/files/'
vector_folder = 'mimic-crx-vectors/'


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


class MimicDataset(Dataset):
    def __init__(self, name, return_image=False, return_label=False, return_report=False, task_binary=False):
        super(MimicDataset, self).__init__()
        assert name in ['train', 'validate', 'test']
        self.name = name
        self.return_image = return_image
        self.return_report = return_report
        self.return_label = return_label
        self.task_binary = task_binary

        # Open split file
        df_samples = pd.read_csv(os.path.join(data_root, split_file))
        df_samples = df_samples.loc[df_samples['split'] == name]
        # Open label file
        df_labels = pd.read_csv(os.path.join(data_root, label_file))
        # Open report file
        df_reports = pd.read_csv(os.path.join(data_root, report_file))

        # Building set
        self.samples = {}
        self.set_file = os.path.join(data_root, name + '_set.pkl')
        if os.path.exists(self.set_file):
            print("Loading " + name + " dataset")
            self.samples = pickle.load(open(self.set_file, 'rb'))
        else:
            print("Building " + name + " dataset")
            excluded = 0
            for index, row in tqdm(df_samples.iterrows(), total=df_samples.shape[0]):
                # Keys are a  triple (subject_id, study_id, dicom_id)
                key = (row['subject_id'], row['study_id'], row['dicom_id'])

                # Fetch label
                label = df_labels.loc[(df_labels['subject_id'] == row['subject_id']) &
                                      (df_labels['study_id'] == row['study_id'])].fillna(0)

                # Fetch report
                report = df_reports.loc[df_reports['study'] == ('s' + str(row['study_id']))]
                if not report['findings'].isna().any():
                    txt = ''.join(report['findings'].values)
                elif not report['impression'].isna().any():
                    txt = ''.join(report['impression'].values)
                else:
                    txt = ''

                if not (True in label.values) or report.empty or txt == '':
                    excluded += 1
                    continue
                self.samples[key] = (label, txt)

            print("Excluded " + str(excluded) + " samples (no label or report). Current sample set is ",
                  len(self.samples))
            # save constructed set
            pickle.dump(self.samples, open(self.set_file, 'wb'))

        self.keys = list(self.samples.keys())
        self.transform = get_transforms(name)

    def __getitem__(self, idx):
        subject_id, study_id, dicom_id = self.keys[idx]
        sample = self.samples[self.keys[idx]]  # A sample is a [label, report]
        img = torch.tensor(0)
        label = torch.tensor(0)
        report = torch.tensor(0)

        if self.return_image:
            img_path = os.path.join(data_root,
                                    image_folder,
                                    'p' + str(subject_id)[:2],  # 10000032 -> p10
                                    'p' + str(subject_id),
                                    's' + str(study_id),
                                    str(dicom_id) + '_256.npy'
                                    )
            try:
                img = self.transform(Image.fromarray(np.load(img_path).astype(np.float32)).convert('RGB'))
            except FileNotFoundError:
                print('image not found for key', self.keys[idx], img_path)
                raise

        if self.return_report:
            _, report = sample

        if self.return_label:
            label, _ = sample
            if self.task_binary:
                if label['No Finding'].all():
                    label = np.array([1, 0])
                else:
                    label = np.array([0, 1])
            else:
                label = (np.array(label.values.tolist()).flatten()[2:] > 0)  # get positive diagnostics
            label = label.astype(np.float)

        return {'idx': idx,
                'key': self.keys[idx],
                'report': report,
                'img': img,
                'label': label}

    def __len__(self):
        return len(self.keys)

# if self.args.model == 'Doc2Vec' or self.args.model == 'DenseNetAux':

# vector_path = os.path.join(data_root,
#                    vector_folder,
#                    str(subject_id) + '-' + str(study_id) + '.npy'
#                    )
#     vector = np.load(vector_path)
