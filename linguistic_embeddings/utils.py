import os
from collections import defaultdict
from tqdm import tqdm
import numpy as np


def get_output_dir(args):
    return os.path.join(args.model_dir, args.model, args.name if args.name is not None else '')


def compute_embeddings(args, model, dataset, outdir, save_vectors=False):
    vectors = defaultdict(list)
    class_names = dataset.task_classes
    print('Computing representations...')
    for sample in tqdm(dataset, total=len(dataset)):
        label = sample['label']
        if sum(label) > 1.0:  # TODO we exclude multilabel, should we ?
            continue
        vector = model(sample)
        if save_vectors:
            np.save(os.path.join(outdir,
                                 "vectors",
                                 str(sample['key'])
                                 ), np.array(vector))

        label = np.where(label == 1.0)[0][0]
        class_name = class_names[label]
        vectors[class_name].append(vector)

    return vectors
