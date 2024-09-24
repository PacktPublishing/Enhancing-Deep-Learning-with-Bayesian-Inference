# Reinforcement Learning Code Example

This example explores the benefit of uncertainty-aware models in the context
of reinforcement learning. The experiments are designed to be run from this
directory.

The `environment` module contains the classes which define the reinforcement
learning environments. There are some modifications moving between the 'standard'
`Environment` class and the `EnvironmentUncertainty` class in order to facilitate
uncertainty-aware behaviour. The environment classes also implement the agent
actions (which is unusual, as this is commonly contained in a separate class, but
  done here for simplicity).

The `models` module contains the classes for the various models: one 'standard'
deep learning model and three different uncertainty-aware models using techniques
from the book.

The `rl_experiment.py` script runs the baseline RL experiment, and the
`rl_experiment_uncertainty.py` script runs the uncertainty-aware experiment.
