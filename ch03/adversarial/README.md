## Data
To download the data used in this example, run
```commandline
mkdir -p data
curl https://images.pexels.com/photos/1317844/pexels-photo-1317844.jpeg > \
data/cat.png
```

```commandline
poetry run python bdl/ch03/adversarial/main.py \
--model-path model.keras
```