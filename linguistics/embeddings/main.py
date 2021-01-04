import sys
import os
import argparse

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from tqdm import tqdm
from collections import defaultdict

from .models import get_foward_function
from .utils import get_output_dir, compute_embeddings
from dataloaders import *
from linguistics.embeddings.train_doc2vec import EpochLogger

parser = argparse.ArgumentParser()
# Dataset
parser.add_argument('--model', type=str, default='Bio_ClinicalBERT',
                    choices=['Bio_ClinicalBERT', 'Doc2Vec', 'BioSentVec', 'BlueBERT', 'CNN'])
parser.add_argument('--model_dir', type=str, default='linguistics/embeddings/models')
parser.add_argument('--dataset', type=str, default="MimicDataset")
parser.add_argument('--dataset_task', type=str, default="all")
parser.add_argument('--task_binary', type=bool, default=False)

# Optional
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--doc2vec_model', type=str, default=None)
parser.add_argument('--cnn_model', type=str, default=None)
parser.add_argument('--save_vectors', type=bool, default=False)
parser.add_argument('--visualization', type=str, default='t-SNE', choices=['t-SNE', 'UMAP'])

args = parser.parse_args()

# Checking arguments are ok
if args.model == 'Doc2Vec':
    assert args.doc2vec_model is not None, 'You need to specifiy a doc2vec model'
if args.model == 'CNN':
    assert args.cnn_model is not None, 'You need to specifiy a CNN model'

# Creating out_dir
outdir = get_output_dir(args)
os.makedirs(os.path.join(outdir, "vectors"), exist_ok=True)
print('Output dir is ', outdir)

# Getting the forward function to compute embeddings
model = get_foward_function(args)

# Getting embeddings according to given dataset
for split in ["validate", "test", "train"]:
    dataset = eval(args.dataset)(split,
                                 return_report=True,
                                 return_label=True,
                                 return_image=args.cnn_model is not None,
                                 task=args.dataset_task)

    vectors = compute_embeddings(args, model, dataset, outdir, save_vectors=args.save_vectors)

    # Computing neighbor embeddings
    # vectors is a dict class_name (string) : vectors [list]

    # Flattening to create training data
    flatten_vectors = [v for _, c in vectors.items() for v in c]
    if args.visualization == 't-SNE':
        # train TSNE
        tsne = TSNE(n_components=2, n_jobs=4, verbose=1, n_iter=2000)
        flatten_repr = tsne.fit_transform(flatten_vectors)
    elif args.visualization == 'UMAP':
        raise NotImplementedError
    else:
        raise NotImplementedError

    # Splitting output per label
    index = 0
    reprs = dict()
    for class_name, _ in vectors.items():
        reprs[class_name] = flatten_repr[index:index + len(vectors[class_name])]
        index += len(vectors[class_name])

    # Plot
    fig = plt.figure()
    c = []
    for class_name, repr in reprs.items():
        c.append(class_name)
        plt.scatter(repr[:, 0], repr[:, 1], s=1.5, alpha=0.4)

    plt.legend(c, markerscale=10, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(args.model + ' ' + args.visualization)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, args.model + '_' + str(split) + '.png'))
    plt.close()
