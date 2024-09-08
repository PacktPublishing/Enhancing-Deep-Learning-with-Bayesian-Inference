# OOD

Download the in-distribution data (cat vs dog):

```commandline
mkdir -p data
curl -X GET https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz data/pets.tgz
tar -xzf data/pets.tgz -C data
```

Then train a model:

```commandline
poetry run python bdl/ch02/ood/train.py
```

Download the OOD data:

```commandline
curl -X GET https://s3.amazonaws.com/fast-ai-imageclas/imagenette-160.tgz \
--output data/imagenette.tgz
tar -xzf data/imagenette.tgz -C data
```

Then, use the model path to test the model on out-of-distribution images

```commandline
poetry run python bdl/ch02/ood/ood.py --model-path model.keras
```
