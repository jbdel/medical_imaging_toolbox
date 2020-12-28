from __future__ import print_function
import os
import numpy as np
from .MimicDataset import MimicDataset


class VectorMimicDataset(MimicDataset):
    def __init__(self, name, vector_folder, **kwargs):
        super(VectorMimicDataset, self).__init__(name, **kwargs)

        self.vector_folder = vector_folder

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        subject_id, study_id, _ = sample['key']
        try:
            vector_path = os.path.join(self.vector_folder,
                                       str(sample['key']) + '.npy'
                                       )
            vector = np.load(vector_path)
        except FileNotFoundError:
            print('Vector not found for key', vector_path)
            raise FileNotFoundError

        sample['vector'] = vector

        return sample


if __name__ == '__main__':
    d = VectorMimicDataset("test", "vector_folder",
                           return_image=True,
                           return_label=True,
                           return_report=True)
    # for _ in tqdm(d):
    #     continue
