import sys
import os
import argparse

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from tqdm import tqdm
from collections import defaultdict

from .models import get_model
from dataloaders.MimicDataset import MimicDataset
from linguistic_embeddings.train_doc2vec import EpochLogger

parser = argparse.ArgumentParser()
# Dataset
parser.add_argument('--model', type=str, default='Bio_ClinicalBERT')
parser.add_argument('--output', type=str, default='linguistic_embeddings')
parser.add_argument('--dataset', type=str, default="MimicDataset")
parser.add_argument('--doc2vec_model', type=str, default=None)


args = parser.parse_args()
outdir = os.path.join(args.output, args.model)
os.makedirs(os.path.join(outdir, "vectors"), exist_ok=True)

model, tokenizer = get_model(args)

for split in ["validate", "test", "train"]:
    dataset = eval(args.dataset)(split, return_report=True, return_label=True)
    class_names = dataset.get_classes_name()

    classes = defaultdict(list)
    study_id_dict = set()
    for sample in tqdm(dataset, total=len(dataset)):
        subject_id, study_id, _ = sample['key']
        label = sample['label']
        if int(study_id) in study_id_dict:
            continue
        if sum(label) > 1.0:  # TODO we exclude multilabel, should we ?
            continue
        study_id_dict.add(int(study_id))

        report = tokenizer(sample['report'])
        vector = model(report)
        np.save(os.path.join(outdir,
                             "vectors",
                             str(subject_id) + '-' + str(study_id)
                             ), np.array(vector))

        idx = np.where(label == 1.0)[0][0]
        classes[idx].append(vector)

    # Flattening to create training data
    flatten_x = [v for _, c in classes.items() for v in c]
    # train TSNE
    tsne = TSNE(n_components=2, n_jobs=4, verbose=1, n_iter=2000)
    flatten_y = tsne.fit_transform(flatten_x)

    # Dividing classes between most and least common
    class_counter = Counter({k: len(v) for k, v in classes.items()})
    num_classes = len(list(classes.keys()))
    half = int(num_classes / 2)
    most = class_counter.most_common(half)
    least = class_counter.most_common()[:-half - 1:-1]

    # Resplitting results per class
    index = 0
    y = dict()
    for i in range(num_classes):
        y[i] = flatten_y[index:index + len(classes[i])]
        index += len(classes[i])

    # Plot
    iterators = [range(num_classes), dict(least).keys(), dict(most).keys()]
    name_iterators = ['all', 'least', 'most']
    for i, (it, name) in enumerate(zip(iterators, name_iterators)):
        fig = plt.figure()
        c = []
        for n in it:
            c.append(class_names[n])
            plt.scatter(y[n][:, 0], y[n][:, 1], s=1.5, alpha=0.4)

        plt.legend(c, markerscale=10, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(args.model + ' t-SNE')
        plt.tight_layout()
        fig.savefig(os.path.join(outdir, "biobert_" + str(split) + "_" + str(name) + ".png"))
        # plt.show()
