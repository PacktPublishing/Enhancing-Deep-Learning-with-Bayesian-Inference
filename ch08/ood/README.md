## Detecting out-of-distribution with MC Dropout

To fit a standard model and a Monte Carlo Dropout model, and see how they behave on a selection of perturbed images, run

```commandline
poetry run python bdl/ch08/ood/main.py --output-dir output
```

You should see that the standard model assigns higher softmax scores to the perturbed images compared to the MC Dropout model.
