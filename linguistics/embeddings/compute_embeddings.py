import sys
import os
import argparse

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap

from .models import get_foward_function
from .utils import get_output_dir, compute_embeddings
from dataloaders import *

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

args = parser.parse_args()

# Creating out_dir
outdir = get_output_dir(args)
os.makedirs(os.path.join(outdir, "vectors"), exist_ok=True)
print('Output dir is', outdir)

# Getting the forward function to compute embeddings
model = get_foward_function(args)

# Getting embeddings according to given dataset
for split in ["val", "test"]:
    dataset: BaseDataset = eval(args.dataset)(split,
                                              return_report=True,
                                              return_label=True,
                                              return_image=args.cnn_model is not None,
                                              task=args.dataset_task)
    dataset_name = type(dataset).__name__

    print('Computing representations for split', split)
    vectors, labels = compute_embeddings(args, model, dataset, outdir, save_vectors=args.save_vectors)

    for visualization in [TSNE(n_components=2, n_jobs=4, verbose=0, n_iter=2000),
                          umap.UMAP(n_neighbors=dataset.num_classes)
                          ]:

        visualization_name = type(visualization).__name__
        print('Computing embeddings using', visualization_name)
        embeddings = visualization.fit_transform(vectors)

        # Plotting
        fig = plt.figure()
        for g in np.unique(labels):
            ix = np.where(labels == g)
            plt.scatter(embeddings[ix, 0], embeddings[ix, 1], s=0.1,
                        cmap='Spectral', label=g)

        plt.legend(markerscale=10, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(args.model + ' ' + visualization_name)
        plt.tight_layout()
        fig.savefig(os.path.join(outdir, args.model
                                 + '_'
                                 + dataset_name
                                 + '_'
                                 + str(split)
                                 + '_'
                                 + visualization_name
                                 + '.png'))
        plt.close()
