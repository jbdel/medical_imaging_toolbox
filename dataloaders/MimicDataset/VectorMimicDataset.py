from __future__ import print_function
import os
import numpy as np
from .MimicDataset import MimicDataset
from linguistics.embeddings.utils import slugify
from tqdm import tqdm


class VectorMimicDataset(MimicDataset):
    def __init__(self, name, vector_folder, **kwargs):
        super(VectorMimicDataset, self).__init__(name, **kwargs)
        assert vector_folder is not None
        self.vector_folder = vector_folder

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        subject_id, study_id, _ = sample['key']
        try:
            vector_path = os.path.join(self.vector_folder,
                                       slugify(sample['key']) + '.npy'
                                       )
            vector = np.load(vector_path)
        except FileNotFoundError:
            raise FileNotFoundError('Vector not found for key', vector_path)

        sample['vector'] = vector

        return sample


if __name__ == '__main__':
    d = VectorMimicDataset("test", "vector_folder",
                           return_image=True,
                           return_label=True,
                           return_report=True)
    for _ in tqdm(d):
        continue
