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
python -m linguistic_embeddings.t_sne --model Bio_ClinicalBERT --output linguistic_embeddings --dataset MimicDataset
python -m linguistic_embeddings.t_sne --model BlueBERT --output linguistic_embeddings --dataset MimicDataset
```
where `dataset` is a dataset from dataloader package.

<b>BioSentVec</b>
Download the pretrained BioSentVec [model](https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioSentVec_PubMed_MIMICIII-bigram_d700.bin) (21GB (700dim, trained on PubMed+MIMIC-III))
```
python -m linguistic_embeddings.t_sne --model BioSentVec --output linguistic_embeddings --dataset MimicDataset
```

<b>Doc2Vec</b>
```
python -m linguistic_embeddings.train_doc2vec --epochs 100 --vector_size 300 --output linguistic_embeddings/Doc2Vec/
python -m linguistic_embeddings.t_sne --model Doc2Vec --output linguistic_embeddings --dataset MimicDataset \
--doc2vec_model DBOW_vector300_window8_count15_epoch100_mimic.doc2vec
```

Using the `linguistic_embeddings.t_sne` command will plot T-SNE and also compute the vector embeddings (saved in npy) in 
the respective model folder.

Every used model is defined in `model.py`

## TODO
BioSentVec and BERT like solution might be trained sentence-wise, not doc wise. Need to compute average of sentences representation.
