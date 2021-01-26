<b>Baseline</b><br/>
Task 6 normal
```
python -m classifier.main --config classifier/configs/cnn.yml
```
```

                precision    recall  f1-score   support

    No Finding       0.81      0.78      0.79      1154
Support Devices      0.64      0.49      0.55       269
      Fracture       0.00      0.00      0.00        24
  Lung Opacity       0.68      0.63      0.66       748
        Cardio       0.60      0.05      0.09       257
       Pleural       0.68      0.62      0.65       373

     micro avg       0.74      0.62      0.67      2825
     macro avg       0.57      0.43      0.46      2825
  weighted avg       0.72      0.62      0.64      2825
   samples avg       0.68      0.64      0.65      2825
```

Task 6 constrained
```
python -m classifier.main --config classifier/configs/cnn_constrained.yml
```
```

                precision    recall  f1-score   support

    No Finding       0.79      0.80      0.79      1154
Support Devices      0.65      0.33      0.44       269
      Fracture       0.00      0.00      0.00        24
  Lung Opacity       0.73      0.54      0.62       748
        Cardio       0.50      0.13      0.21       257
       Pleural       0.65      0.66      0.66       373

     micro avg       0.74      0.60      0.66      2825
     macro avg       0.55      0.41      0.45      2825
  weighted avg       0.71      0.60      0.64      2825
   samples avg       0.65      0.63      0.63      2825
```
We can say that adding cosine loss doesnt really deteriorate resuls

Task all (need to compare against hidden strat)
```
python -m classifier.main --config classifier/configs/cnn_all.yml
```

```
                            precision    recall  f1-score   support

               Atelectasis       0.46      0.16      0.24       279
              Cardiomegaly       0.47      0.06      0.11       236
             Consolidation       0.00      0.00      0.00        50
                     Edema       0.62      0.04      0.08       185
Enlarged Cardiomediastinum       0.00      0.00      0.00        30
                  Fracture       0.00      0.00      0.00        24
               Lung Lesion       0.00      0.00      0.00        56
              Lung Opacity       0.46      0.20      0.28       328
                No Finding       0.79      0.79      0.79      1154
          Pleural Effusion       0.67      0.59      0.63       334
             Pleural Other       0.00      0.00      0.00         6
                 Pneumonia       0.29      0.01      0.03       139
              Pneumothorax       0.44      0.16      0.24        50
           Support Devices       0.61      0.43      0.50       269

                 micro avg       0.70      0.43      0.54      3140
                 macro avg       0.34      0.18      0.21      3140
              weighted avg       0.60      0.43      0.47      3140
               samples avg       0.58      0.52      0.54      3140
```

Hidden strat (10 neighbours) on totally correct predictions: 
```
python -m classifier.main --ckpt classifier/checkpoints/my_model_constrained/best0.6354023763385523.pkl -o metrics=[HiddenStratMetric,ClassificationMetric]```
```
```
                            precision    recall  f1-score   support

               Atelectasis       0.96      0.56      0.71        78
              Cardiomegaly       1.00      1.00      1.00         8
             Consolidation       0.83      0.26      0.40        19
                     Edema       0.73      0.48      0.58        46
Enlarged Cardiomediastinum       0.00      0.00      0.00         0
                  Fracture       0.00      0.00      0.00         0
               Lung Lesion       0.75      0.25      0.38        12
              Lung Opacity       0.70      0.80      0.75        82
                No Finding       1.00      1.00      1.00       886
          Pleural Effusion       0.96      0.71      0.81       102
             Pleural Other       0.00      0.00      0.00         0
                 Pneumonia       0.71      0.35      0.47        34
              Pneumothorax       1.00      0.11      0.20         9
           Support Devices       1.00      1.00      1.00        26

                 micro avg       0.96      0.88      0.92      1302
                 macro avg       0.69      0.47      0.52      1302
              weighted avg       0.95      0.88      0.90      1302
               samples avg       0.95      0.94      0.94      1302
```

Hidden strat (10 neighbours) on all predictions: 
```
python -m classifier.main --ckpt classifier/checkpoints/my_model_constrained/best0.6354023763385523.pkl -o metrics=[HiddenStratMetric,ClassificationMetric]```
```
```
                            precision    recall  f1-score   support

               Atelectasis       0.83      0.33      0.47       279
              Cardiomegaly       0.93      0.11      0.19       236
             Consolidation       0.58      0.14      0.23        50
                     Edema       0.87      0.38      0.53       185
Enlarged Cardiomediastinum       0.00      0.00      0.00        30
                  Fracture       0.00      0.00      0.00        24
               Lung Lesion       0.67      0.07      0.13        56
              Lung Opacity       0.61      0.39      0.48       328
                No Finding       0.79      0.80      0.79      1154
          Pleural Effusion       0.89      0.50      0.64       334
             Pleural Other       0.00      0.00      0.00         6
                 Pneumonia       0.67      0.16      0.26       139
              Pneumothorax       0.82      0.18      0.30        50
           Support Devices       0.65      0.33      0.44       269

                 micro avg       0.78      0.49      0.60      3140
                 macro avg       0.59      0.24      0.32      3140
              weighted avg       0.76      0.49      0.56      3140
               samples avg       0.61      0.56      0.58      3140
```
Overall, an improvement on the weighted f1 score of 9%.
We can notice improvements for Atelectasis, Lung Lesion, Pneumonia, Edema (! from 0.08 to 0.53 !), and so forth.