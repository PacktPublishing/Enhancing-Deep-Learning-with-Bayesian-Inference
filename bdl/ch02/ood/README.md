# OOD

Download the data:

```commandline
curl -X GET https://s3.amazonaws.com/fast-ai-imageclas/imagenette-160.tgz \
--output data/imagenette.tgz
tar -xzf data/imagenette.tgz -C data
```

Then train a model:

```commandline
poetry run python bdl/ch02/ood/train.py
```

Then, use the model path to test the model on out-of-distribution images

```commandline
poetry run python bdl/ch02/ood/ood.py --model-path bdl/ch02/ood/model.keras
```
