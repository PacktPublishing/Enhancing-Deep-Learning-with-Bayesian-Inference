## Active learning

Code example for active learning using different sources of uncertainty, in particular knowledge/epistemic uncertainty.

To run the code, use:

```commandline
poetry run python ch08/active_learning/main.py --acquisition-type random
```

To train a model iteratively with more and more data randomly selected.

Then, for the comparison, run

```commandline
poetry run python ch08/active_learning/main.py --acquisition-type knowledge_uncertainty
```

To train a model where data is selected via knowledge/epistemic uncertainty.

Finally, you can plot the differences between the two methods with

```commandline
poetry run python ch08/active_learning/plot.py \
--uuid1 {uuid_1} --acq1 random \
--uuid2 {uuid_2} --acq2 knowledge_uncertainty
```

To plot the accuracy difference between the two methods. Replace {uuid_1} and {uuid_2} with the UUIDs of the uuids of the directories created by the previous steps.

You should see that selecting data via knowledge uncertainty gives better performance with fewer samples, and a higher final accuracy.
In addition, you can inspect the images that were selected by both methods, and should see that the
images selected via knowledge uncertain are more diverse.
