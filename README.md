Python3.8
<p><b>linguistic_embedding package</b></p>

This package is used to compute vector representations of a medical dataset reports. So far, one can use [BioclinicalBERT](https://github.com/EmilyAlsentzer/clinicalBERT/), [BlueBert](https://github.com/ncbi-nlp/bluebert/)
, [BioSentVec](https://github.com/ncbi-nlp/BioSentVec) and Doc2Vec.

To run a model, let's use:
```
python -m linguistics.embeddings.main \
    --model BioSentVec \
    --model_dir linguistics/embeddings/models  \
    --dataset MimicDataset  \
    --dataset_task binary  \
    --save_vectors True \
    --visualization t-SNE
```
where dataset argument takes a dataset from the package dataloaders 
(see dedicated [README](https://github.com/jbdel/medical_imaging_toolbox/tree/main/dataloaders)).

The `dataset_task` argument defines how the task is configured. For example, we want to use all the label, or only binary, or regroup 
some labels into clusters. Label trees can be defined as show 
for the [MimicDataset](https://github.com/jbdel/medical_imaging_toolbox/tree/main/dataloaders/MimicDataset/BaseMimic.py).

Vectors are created in `linguistics/embeddings/models/BioSentVec/vectors/` and t-SNE plots are dump 
also (see `visualization` command):

<img src='https://i.imgur.com/tT7h3hb.png' width="400px" /><br/>
<i>MimicDataset test-set</i>

Above is the results of t-SNE on the six-class variant (`--dataset_task six`) of MimicDataset.
<p><b>classifier package</b></p>

It is also possible to use the classifier package to run a model on any medical dataset. So far, only CNN are available. Use 
the following command to train a model:

```
python -m classifier.main \
     --backbone densenet169 \
     --name my_model \
     --output classifier/checkpoints \
     --dataset MimicDataset \
     --dataset_task binary \
     --losses classification \
     --early_stop_metric f1_score_weighted \
     --use_scheduler True \
     --return_image True \
     --return_label True 
```

Or you can constrain the CNN to fit the representation learned using the linguistic_embedding package:
```
python -m classifier.main \
 --backbone densenet169 \
 --name my_model \
 --output classifier/checkpoints \
 --dataset VectorMimicDataset \
 --losses classification cosine \
 --vector_size 700 \
 --vector_folder linguistics/embeddings/models/BioSentVec/vectors/ \
 --dataset_task binary \
 --return_image True \
 --return_label True 
```

You can also plot and save the vectors of any clssifier in the linguistic_embedding package

<p><b>dataloaders package</b></p>

So far is available the MimicDataset.
