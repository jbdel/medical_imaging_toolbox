To run a model on MimicDataset using a resnet18 backbone, use the following command
```
python -m classifier.main --model resnet18 \
 --name resnet18_baseline \
 --output classifier/checkpoints \
 --dataset MimicDataset \
 --early_stop_metric f1_score_weighted \
 --losses classification \
 --num_classes 2 \
 --pred_func amax \
 --lr_base 0.01 \
 --use_scheduler True \
 --return_image True \
 --return_label True
```

To constrain a model on vectors, use:

```
python -m classifier.main --model resnet18 \
 --name resnet18_baseline \
 --output classifier/checkpoints \
 --dataset #TODO \
 --early_stop_metric f1_score_weighted \
 --losses classification cosine \
 --num_classes 2 \
 --pred_func amax \
 --lr_base 0.01 \
 --use_scheduler True \
 --return_image True \
 --return_label True
 --return_vector True
```