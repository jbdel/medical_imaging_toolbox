# R2Gen

This is the implementation of [Generating Radiology Reports via Memory-driven Transformer](https://arxiv.org/pdf/2010.16056.pdf) at EMNLP-2020.

Original link [here](https://github.com/cuhksz-nlp/R2Gen)


### Download R2Gen
You can download the models we trained for each dataset from [here](https://github.com/cuhksz-nlp/R2Gen/blob/main/data/r2gen.md).

### Datasets
We use two datasets (IU X-Ray and MIMIC-CXR) in our paper.

For `IU X-Ray`, you can download the dataset from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing) and then put the files in `data/iu_xray`.

For `MIMIC-CXR`, you can download the dataset from [here](https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing) and then put the files in `data/mimic_cxr`.

### Run on IU X-Ray

Run `bash run_iu_xray.sh` to train a model on the IU X-Ray data.

### Run on MIMIC-CXR

Run `bash run_mimic_cxr.sh` to train a model on the MIMIC-CXR data.

### Extract reports

`python extract.py`

This will create a r2gen_mimic.json file with ground truth and hypotheses. 
