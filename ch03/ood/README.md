### OOD
To download the data, execute the following commands in the root directory of the project:
## data

For the out-of-distribution example, you need two datasets that can be downloaded as follows:
```commandline
mkdir -p data
curl -X GET https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz \
--output data/pets.tgz
tar -xzf data/pets.tgz -C data
curl -X GET https://s3.amazonaws.com/fast-ai-imageclas/imagenette-160.tgz \
--output data/imagenette.tgz
tar -xzf data/imagenette.tgz -C data
```
