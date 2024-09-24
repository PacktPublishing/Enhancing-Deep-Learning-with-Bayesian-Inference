import numpy as np
import tensorflow as tf
from scipy.spatial.distance import euclidean
from tensorflow.keras import Model, Sequential, layers, optimizers, metrics, losses
import pandas as pd
from sklearn.preprocessing import StandardScaler
import copy

from environment import EnvironmentUncertainty
from models import RLModelDropout

env_size = 8
state_size = 6
n_actions = 4
epsilon = 1.0
history = {"state": [], "reward": []}
model = RLModelDropout(state_size, n_actions, num_epochs=200)
n_samples = 1000
max_steps = 500
regrets = []

collisions = 0

for i in range(60):
    if i < 20:
        env = EnvironmentUncertainty(env_size, max_steps=max_steps)
        obstacle = False
    else:
        obstacle = True
        epsilon = 0
        env = EnvironmentUncertainty(env_size, max_steps=max_steps, obstacle=True)
    while not env.is_done:
        state = env.get_state()
        obstacle_proximity = env.get_obstacle_proximity()
        if np.random.rand() < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = model.predict(state, obstacle_proximity, obstacle)
        # print(action)
        # print(env.delta)
        reward = env.update(action)
        history["state"].append(np.concatenate([state, [action], [obstacle_proximity[action]]]))
        history["reward"].append(reward)
    if env.total_steps == max_steps:
        print(f"Failed to find target for episode {i}. Epsilon: {epsilon}")
    elif env.total_steps < env.ideal_steps:
        print(f"Collided with obstacle during episode {i}. Epsilon: {epsilon}")
        collisions += 1
    else:
        print(f"Completed episode {i} in {env.total_steps} steps. Ideal steps: {env.ideal_steps}. Epsilon: {epsilon}")
    regrets.append(np.abs(env.total_steps-env.ideal_steps))
    idxs = np.random.choice(len(history["state"]), n_samples)
    if not obstacle:
        model.fit(np.array(history["state"])[idxs], np.array(history["reward"])[idxs])
        epsilon-=epsilon/10
