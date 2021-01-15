import sys
import os
import argparse

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap

from .utils import slugify
from tqdm import tqdm
from dataloaders import *
from omegaconf import OmegaConf
from .models import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    # Create output directory
    cfg.outdir = os.path.join(cfg.experiment.output_dir, cfg.experiment.name)
    os.makedirs(os.path.join(cfg.outdir, "vectors"), exist_ok=True)
    print('Output dir is', cfg.outdir)

    # Getting model
    model = eval(cfg.model.name)(cfg)
    print('Using model', type(model).__name__)

    for split in cfg.experiment.plot_split:
        # Getting embeddings
        print('Computing representations for split', split)

        dataset: BaseDataset = eval(cfg.dataset.name)(split,
                                                      return_report=True,
                                                      return_label=True,
                                                      return_image=cfg.dataset.return_image is not None,
                                                      task=cfg.dataset.task)

        vectors, labels = list(), list()
        for sample in tqdm(dataset, total=len(dataset)):
            label = sample['label']
            vector = model(sample)
            if cfg.experiment.save_vectors:
                np.save(os.path.join(cfg.outdir,
                                     "vectors",
                                     slugify(sample['key'])
                                     ), np.array(vector))

            # TODO we exclude multilabel samples for plotting, should we ?
            if sum(label) > 1.0:
                continue

            c = np.where(label == 1.)[0][0]
            labels.append(dataset.task_classes[c])
            vectors.append(vector)

        labels = np.array(labels)
        vectors = np.array(vectors)

        # Plotting visualization
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
            plt.title(cfg.model.name + ' ' + visualization_name)
            plt.tight_layout()
            fig.savefig(os.path.join(cfg.outdir, cfg.experiment.name
                                     + '_'
                                     + str(split)
                                     + '_'
                                     + visualization_name
                                     + '.png'))
            plt.close()
