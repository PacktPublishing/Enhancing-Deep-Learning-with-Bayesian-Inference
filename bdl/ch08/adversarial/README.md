# Adversarial robustness

Download the data

```commandline
mkdir -p data
curl -X GET https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz data/pets.tgz
tar -xzf data/pets.tgz -C data
```

Then run

```commandline
poetry run python bdl/ch08/adversarial/main.py
```

to train a neural network model for image classification, perform
adversarial attacks using the Fast Gradient Sign Method (FGSM), and evaluate the model's performance using both standard predictions and Monte Carlo dropout.
