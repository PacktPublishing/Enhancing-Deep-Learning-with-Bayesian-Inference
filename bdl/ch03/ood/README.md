### OOD
To download the data, execute the following commands in the root directory of the project:
## data

For the out-of-distribution example, you need two datasets that can be downloaded as follows:
```commandline
mkdir -p data/ch03/ood
curl -X GET https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz \
--output data/ch03/ood/pets.tgz
tar -xzf data/ch03/ood/pets.tgz -C data/ch03/ood
curl -X GET https://s3.amazonaws.com/fast-ai-imageclas/imagenette-160.tgz \
--output data/ch03/ood/imagenette.tgz
tar -xzf data/ch03/ood/imagenette.tgz -C data/ch03/ood
```

Then run the code with
```commandline
poetry run python ch03/ood/main.py
```