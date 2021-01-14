To put in the data folder:
https://drive.google.com/drive/folders/1pU97NrwdqG9raBm4aXx4gep2FfUFE_Rp?usp=sharing

Python3.8
<p><b>linguistic_embedding package</b></p>

This package is used to compute vector representations of a medical dataset reports. So far, one can use [BioclinicalBERT](https://github.com/EmilyAlsentzer/clinicalBERT/), [BlueBert](https://github.com/ncbi-nlp/bluebert/)
, [BioSentVec](https://github.com/ncbi-nlp/BioSentVec) and Doc2Vec.

To run a model, let's use:
```
python -m linguistics.embeddings.compute_embeddings --config BioSentVec
```

(see dedicated config folder for more examples)
For any config, you can set save_vectors to True.

<img src='https://i.imgur.com/tT7h3hb.png' width="400px" /><img src='https://i.imgur.com/XAr6uDH.png' width="400px" /><br/>
<br/>
<i>MimicDataset test-set and eval-set</i>

Above is the results of t-SNE on the six-class variant of MimicDataset.
<p><b>classifier package</b></p>

It is also possible to use the classifier package to run a model on any medical dataset. So far, only CNN are available. Use 
the following command to train a model:

To run a model on MimicDataset using a densenet backbone, use the following command
```
python -m classifier.main --config classifier/configs/cnn.yml
```

