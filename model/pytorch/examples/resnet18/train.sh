dffml train \
  -model resnet18 \
  -model-clstype str \
  -model-classifications ants bees \
  -model-directory resnet18_model \
  -model-epochs 5 \
  -model-batch_size 32 \
  -model-useCUDA \
  -model-features image:int:$((500*500)) \
  -model-predict label:str:1 \
  -sources f=dir \
    -source-foldername hymenoptera_data/train \
    -source-feature image \
    -source-labels ants bees \
  -log debug