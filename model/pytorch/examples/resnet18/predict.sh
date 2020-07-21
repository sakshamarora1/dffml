dffml predict all \
  -model resnet18 \
  -model-clstype str \
  -model-classifications ants bees \
  -model-directory resnet18_model \
  -model-useCUDA \
  -model-features image:int:$((500*500)) \
  -model-predict label:str:1 \
  -sources f=csv \
    -source-filename unknown_images.csv \
    -source-loadfiles image \
  -log debug \
  -pretty