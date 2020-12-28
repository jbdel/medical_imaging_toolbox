To run a model on MimicDataset using a resnet18 backbone, use the following command
```
for arch in densenet169
do
    python -m classifier.main \
     --backbone densenet169 \
     --name my_model \
     --output classifier/checkpoints \
     --dataset MimicDataset \
     --dataset_task binary \
     --return_image True \
     --return_label True \
     --losses classification \
     --early_stop_metric f1_score_weighted \
     --pred_func amax \
     --lr_base 0.01 \
     --batch_size 64 \
     --use_scheduler True
done
```

To constrain a model on vectors, use:

```
todo
```