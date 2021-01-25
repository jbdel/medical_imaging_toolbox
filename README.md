<b>Mimic-crx</b>

To put in the data/mimic-crx folder, images and annotations.json from [here](https://drive.google.com/drive/folders/1pU97NrwdqG9raBm4aXx4gep2FfUFE_Rp?usp=sharing)

annotations.json contains all dataset informations:
```
{"val":
     [{'id': str,
       'study_id': str,
       'subject_id': str,
       'image_path': str,
       'split': str,
       'label': list,
       'report'dict}, ...],
 "train": [{...}, ...],
 "test": [{...}, ...]
 }
```
The report key for each sample is a dictionary:
```
{'findings': str,
 'impression': str,
 'background': str,
 'MIT-LCP_findings': str,
 'MIT-LCP_impression': str,
 'MIT-LCP_last_paragraph': str,
 'MIT-LCP_comparison': str,
 'r2gen'str}
```
findings, impression and background are extracted from [here](https://github.com/abachaa/MEDIQA2021/tree/main/Task3)

MIT-LCP are extracted from physionet official account ([MIT Laboratory for Computational Physiology](https://github.com/MIT-LCP/mimic-cxr/tree/master/txt)).
These ones seems to work best.

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

#TODO

- Benchmark a baseline
- Improve embeddings
- Implement embeddings eval (of some sort)
- retrain R2Gen
- Incorporate VisualBert
- Implement 'find nearest report' at test time
- Implement GDRO
- Publish a paper and get rich$$
