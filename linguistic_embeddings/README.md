### Linguistic embeddings
Requirements
```
pip install pandas, sklearn, matplotlib, tqdm, transformers, nltk, gensim
```

To install sent2vec

```
cd BioSentVec/sent2vec
pip install Cython
pip install .
```

This package computes the embeddings of reports. We evaluate [BioclinicalBERT](https://github.com/EmilyAlsentzer/clinicalBERT/), 
[BlueBert](https://github.com/ncbi-nlp/bluebert/), [BioSentVec](https://github.com/ncbi-nlp/BioSentVec) and Doc2Vec.

<b>BioClinicalBERT</b> and <b>BlueBert</b> are directly available through the [HuggingFace transformers](https://github.com/huggingface/transformers) library.
```
python -m linguistic_embeddings.main --model {Bio_ClinicalBERT,Bio_ClinicalBERT} \
                                     --model_dir linguistic_embeddings/models  \
                                     --dataset MimicDataset  \
                                     --dataset_task binary  \
                                     --visualization t-SNE
```
where `dataset` is a dataset from dataloader package.

<b>BioSentVec</b>
Download the pretrained BioSentVec [model](https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioSentVec_PubMed_MIMICIII-bigram_d700.bin) (21GB (700dim, trained on PubMed+MIMIC-III))
```
python -m linguistic_embeddings.main \
    --model BioSentVec \
    --model_dir linguistic_embeddings/models  \
    --dataset MimicDataset  \
    --dataset_task binary  \
    --save_vectors True \
    --visualization t-SNE
```

<b>Doc2Vec</b>
```
python -m linguistic_embeddings.main \
    --model Doc2Vec \
    --model_dir linguistic_embeddings/models  \
    --dataset MimicDataset  \
    --dataset_task binary  \
    --doc2vec_model mymodel/DBOW_vector300_window8_count15_epoch100_mimic.doc2vec \
    --name mymodel \
    --save_vectors True \
    --visualization t-SNE
```

<b>CNN</b>
To get the vectors of a model trained in the classifier package, use:
```
python -m linguistic_embeddings.main \
    --model CNN \
    --model_dir linguistic_embeddings/models  \
    --dataset MimicDataset  \
    --dataset_task six  \
    --cnn_model classifier/checkpoints/densenet169/best196.pkl  \
    --name densenet169 \
    --visualization t-SNE
```
Every used model is defined in `model.py`

## TODO
BioSentVec and BERT like solution might be trained sentence-wise, not doc wise. Need to compute average of sentences representation.
