<b>Setting up environment</b>

Make sure you are in the top-level directory. Create a virtual environment by running the following command:
```
python3 -m venv envs/medical_toolbox
```

Next, activate the environment:
```
source envs/medical_toolbox/bin/activate
```

Install all the necessary modules:
```
python -m pip install -r requirements.txt
```

Finally, install sent2vec from their github page:
```
https://github.com/epfml/sent2vec
```
Note: on Mac it was necessary to add a MACOSX_DEPLOYMENT_TARGET flag when running pip install, as follows:
```
MACOSX_DEPLOYMENT_TARGET=10.14 pip install . (assuming you are in the sent2vec folder cloned from the GitHub)
```

<b>Mimic-crx</b>

The data used in this project. Please put all the files into a folder called data/mimic-crx. Download the images and annotations (stored in an annotations.json file) [here](https://drive.google.com/drive/folders/1pU97NrwdqG9raBm4aXx4gep2FfUFE_Rp?usp=sharing)

annotations.json contains all the information for a given piece of data and is divided up into train, test and validation splits:
```
{"val":
     [{'id': str,
       'study_id': str,
       'subject_id': str,
       'image_path': str,
       'split': str,
       'label': list,
       'report': dict[see below]}, ...],
 "train": [{...}, ...],
 "test": [{...}, ...]
 }
```
The report dictionary contains the following information:
```
{'findings': str,
 'impression': str,
 'background': str,
 'MIT-LCP_findings': str,
 'MIT-LCP_impression': str,
 'MIT-LCP_last_paragraph': str,
 'MIT-LCP_comparison': str,
 'r2gen': str}
```
findings, impression and background were extracted from [here](https://github.com/abachaa/MEDIQA2021/tree/main/Task3)

MIT-LCP information was extracted from physionet official account ([MIT Laboratory for Computational Physiology](https://github.com/MIT-LCP/mimic-cxr/tree/master/txt)). These seemed to work best in our initial experiments, however we have provided the original MIMIC findings, impression and background as well.

R2gen is the ouput of the transformer module. This one might need retraining.

<p><b>linguistic_embedding package</b></p>

This package is used to compute vector representations of a medical dataset reports. So far, one can use [BioclinicalBERT](https://github.com/EmilyAlsentzer/clinicalBERT/), [BlueBert](https://github.com/ncbi-nlp/bluebert/)
, [BioSentVec](https://github.com/ncbi-nlp/BioSentVec) and Doc2Vec.

To run a model, let's use:
```
python -m linguistics.embeddings.compute_embeddings --config linguistics/embeddings/configs/biosentvec.yml
```

To train doc2vec, use:
```
python -m linguistics.embeddings.compute_embeddings --config linguistics/embeddings/configs/doc2vec_train.yml
```

<p><b>classifier package</b></p>

It is possible to use the classifier package to run a model on any medical dataset. So far, only 
CNN are available.
To run a model on MimicDataset using a densenet backbone, use the following command
```
python -m classifier.main --config classifier/configs/cnn.yml
```
To run a model on MimicDataset using a densenet backbone and constraining output on vectors,
use the following command
```
python -m classifier.main --config classifier/configs/cnn_constrained.yml 
```

<p><b>Make an experiment end to end</b></p>
Using:

```
python -m linguistics.embeddings.compute_embeddings --config linguistics/embeddings/configs/doc2vec_train.yml
```
will train a doc2vec model with the MIT-LCP sections using the top_section_MIT-LCP policy 
(see linguistics/embeddings/utils.py for more information).

Output is `linguistics/embeddings/output/doc2vec_mimic_mit` and vectors are in 
`linguistics/embeddings/output/doc2vec_mimic_mit/vectors`

You can then run a constrained cnn using

```
python -m classifier.main --config classifier/configs/cnn_constrained.yml 
```
(make sure that `vector_file` is set to the right path in the config file)

# Current TODO

- Improve embeddings
- Integrate IU dataset
- Implement GDRO
- Publish a paper and get rich $$

# Long-term TODO

- Incorporate VisualBert

# Deadlines

- EMNLP - May 10
- NeurIps - May 20
