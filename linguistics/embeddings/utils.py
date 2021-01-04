import osembeddings
import unicodedata
import re
from collections import defaultdict
from tqdm import tqdm
import numpy as np


def slugify(value):
    """
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    https://github.com/django/django/blob/master/django/utils/text.py#L394
    """
    value = str(value)
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')

def get_output_dir(args):
    return os.path.join(args.model_dir, args.model, args.name if args.name is not None else '')


def compute_embeddings(args, model, dataset, outdir, save_vectors=False):
    vectors = defaultdict(list)
    class_names = dataset.task_classes
    print('Computing representations...')
    for sample in tqdm(dataset, total=len(dataset)):
        label = sample['label']
        vector = model(sample)
        if save_vectors:
            np.save(os.path.join(outdir,
                                 "vectors",
                                 slugify(sample['key'])
                                 ), np.array(vector))

        if sum(label) > 1.0:  # TODO we exclude multilabel for plotting, should we ?
            continue
        label = np.where(label == 1.0)[0][0]
        class_name = class_names[label]
        vectors[class_name].append(vector)

    return vectors
